"""CPU-only R202 correctness checks; no GPU or experimental accuracy result."""
import argparse
import copy
import gc
import json
import math
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from research.common import ROOT, CLASSES, sha256, write_json
from . import convnext_core as core
from . import prepare_protocol as protocol
from .train_convnext_base import (atomic_checkpoint, runtime_versions, build_payload,
                                 checked_resume, select_winners, selected, WINNERS)

OUTPUT = ROOT / 'results/image_level_replication/v2/preparation'


def weighted_accumulation_checks(microbatch):
    """Compare unequal microbatch weights and final partial groups analytically."""
    generator = torch.Generator().manual_seed(102)
    weights = torch.tensor([.25, .75, 1., 1.3, 2., 3., 4.], dtype=torch.float64)
    for size in (17, 32, 15, 1):
        logits = torch.randn(size, 7, generator=generator, dtype=torch.float64)
        labels = torch.arange(size) % 7
        full = logits.clone().requires_grad_(True)
        expected = F.cross_entropy(full, labels, weight=weights)
        expected.backward()
        accumulated = logits.clone().requires_grad_(True)
        denominator = weights[labels].sum()
        total = 0.
        for offset in range(0, size, microbatch):
            stop = min(size, offset + microbatch)
            loss = F.cross_entropy(accumulated[offset:stop], labels[offset:stop],
                                   weight=weights, reduction='sum') / denominator
            total += float(loss.detach())
            loss.backward()
        assert torch.allclose(full.grad, accumulated.grad, rtol=1e-12, atol=1e-12)
        assert abs(total - float(expected.detach())) < 1e-12
    return True


def meaningful_counter_checks(c):
    flat = [dict(epoch=epoch, val_accuracy=.85 + (epoch % 2) * .0004,
                 val_macro_f1=.8 + (epoch % 3) * .0004) for epoch in range(1, 51)]
    assert not core.meaningful(flat[:40], c)['stop']
    final = core.meaningful(flat, c)
    assert final['stop'] and final['reason'] == 'maximum_epochs'
    assert final['meaningful_improvement_epochs'] == [1] and final['stale'] == 49
    accumulating = [dict(epoch=epoch, val_accuracy=.85 + (epoch - 1) * .0005,
                         val_macro_f1=.8) for epoch in range(1, 7)]
    assert core.meaningful(accumulating, c)['meaningful_improvement_epochs'] == [1, 5]
    f1_rise = [dict(epoch=epoch, val_accuracy=.85,
                   val_macro_f1=.8 + (.003 if epoch >= 4 else 0.)) for epoch in range(1, 8)]
    assert core.meaningful(f1_rise, c)['meaningful_improvement_epochs'] == [1, 4]
    return True


def assert_state_equal(left, right):
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def synthetic_resume_checks(c, sig):
    """Exercise committed warmup/full-stage schema with tiny disposable states."""
    model = nn.Linear(2, 7)
    opt = torch.optim.AdamW([dict(params=list(model.parameters()), group_name='head_decay',
                                  stage='head', base_lr=c['head_lr'], lr=c['head_lr'],
                                  weight_decay=c['weight_decay'])], foreach=False)
    history, winners = [], dict.fromkeys(WINNERS)
    ema, ema_updates = None, 0
    fixtures = ((.85, .8), (.851, .799), (.851, .805))
    for epoch, (accuracy, f1) in enumerate(fixtures, start=1):
        core.apply_learning_rates(opt, c, epoch)
        opt.zero_grad(set_to_none=True)
        model(torch.ones(1, 2)).square().sum().backward()
        opt.step()
        raw = dict(branch='raw', epoch=epoch,
                   metrics=dict(accuracy=accuracy, macro_f1=f1), predictions=[])
        ema_record = None
        if epoch > 2:
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            core.update_ema(ema, model, c['ema']['decay'])
            ema_updates += 1
            ema_record = dict(branch='ema', epoch=epoch,
                              metrics=dict(accuracy=accuracy, macro_f1=f1), predictions=[])
        records = dict(raw=raw, ema=ema_record)
        winners = select_winners(winners, records, dict(raw=model, ema=ema))
        history.append(dict(epoch=epoch, accepted_updates=1,
                            val_accuracy=accuracy, val_macro_f1=f1,
                            raw_val_accuracy=accuracy, raw_val_macro_f1=f1,
                            ema_val_accuracy=accuracy if ema_record else None,
                            ema_val_macro_f1=f1 if ema_record else None))
        saved = build_payload(c, sig, model, opt, ema, epoch, history, winners,
                              records, epoch, ema_updates, [], 0., cuda_rng=False)
        checked_resume(saved, c, sig, require_cuda_rng=False)
    assert selected(winners)['epoch'] == 2 and selected(winners)['branch'] == 'raw'
    assert selected(winners, 'macro_f1')['epoch'] == 3 and selected(winners, 'macro_f1')['branch'] == 'raw'
    assert winners['raw_accuracy']['epoch'] == 2
    corruptions = []
    for name in ('updates', 'ema_updates', 'history', 'class_order', 'outer_scope', 'winner', 'lr', 'nonfinite'):
        altered = copy.deepcopy(saved)
        if name == 'updates':
            altered['optimizer_updates'] += 1
        elif name == 'ema_updates':
            altered['ema_updates'] += 1
        elif name == 'history':
            altered['history'][0]['epoch'] = 2
        elif name == 'class_order':
            altered['class_order'].reverse()
        elif name == 'outer_scope':
            altered['outer_assessment_loaded'] = True
        elif name == 'winner':
            altered['winners']['raw_accuracy']['epoch'] = 3
        elif name == 'lr':
            altered['optimizer']['param_groups'][0]['lr'] *= .5
        elif name == 'nonfinite':
            altered['model']['weight'][0, 0] = float('nan')
        try:
            checked_resume(altered, c, sig, require_cuda_rng=False)
        except (ValueError, FloatingPointError):
            corruptions.append(name)
        else:
            raise AssertionError('Corrupted resume accepted: ' + name)
    try:
        checked_resume(saved, c, sig, require_cuda_rng=True)
    except ValueError:
        corruptions.append('cuda_rng_required_for_real_resume')
    else:
        raise AssertionError('CPU probe accepted as a CUDA resume')
    return dict(three_stage_synthetic_schema_checked=True, earliest_epoch_raw_tie_selection_checked=True,
                corrupted_resume_rejections=corruptions,
                note='Handcrafted selector metrics are synthetic schema fixtures; they are not model or project results.')


def ema_buffer_checks(c):
    """EMA averages parameters and copies buffers, including nonfloat buffers."""
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1., 2.]))
            self.register_buffer('running_marker', torch.tensor([3., 4.]))
            self.register_buffer('step_marker', torch.tensor(0, dtype=torch.long))
    current = Tiny()
    ema = copy.deepcopy(current).requires_grad_(False)
    with torch.no_grad():
        current.weight.add_(1.)
        current.running_marker.add_(2.)
        current.step_marker.add_(1)
    decay = c['ema']['decay']
    core.update_ema(ema, current, decay)
    assert torch.allclose(ema.weight, torch.tensor([1., 2.]) + (1. - decay))
    assert torch.equal(ema.running_marker, current.running_marker)
    assert torch.equal(ema.step_marker, current.step_marker)
    return True


def optimizer_checks(model, opt, c):
    named = dict(model.named_parameters())
    parameter_names = {id(parameter): name for name, parameter in named.items()}
    covered = []
    for group in opt.param_groups:
        assert len(group['params']) > 0
        for parameter in group['params']:
            name = parameter_names[id(parameter)]
            covered.append(name)
            no_decay = parameter.ndim == 1 or name.endswith('.bias')
            assert group['weight_decay'] == (0. if no_decay else c['weight_decay']), name
    assert len(covered) == len(set(covered)) == len(named)
    assert set(covered) == set(named)
    observed = {}
    for epoch in (1, 2, 3, 4, 5, 6, 50):
        rates = core.apply_learning_rates(opt, c, epoch)
        observed[str(epoch)] = rates
        for group in opt.param_groups:
            assert math.isfinite(group['lr']) and group['lr'] >= 0.
            names = [parameter_names[id(parameter)] for parameter in group['params']]
            def expected_lr(name):
                is_head = name.startswith('head.')
                if epoch <= 2:
                    return c['head_lr'] if is_head else 0.
                if epoch <= 5:
                    factor = 1. if is_head else (epoch - 2) / 3.
                else:
                    floor = c['scheduler']['minimum_factor']
                    factor = floor + (1. - floor) * .5 * (1. + math.cos(math.pi * (epoch - 5) / 45.))
                if is_head:
                    return c['head_lr'] * factor
                if name.startswith('stem.'):
                    stage = 'stem'
                elif name.startswith('stages.'):
                    stage = 'stage' + name.split('.')[1]
                else:
                    raise AssertionError('Unrecognized backbone parameter: ' + name)
                return c['backbone_lr'] * c['stage_lr_multipliers'][stage] * factor
            assert all(math.isclose(group['lr'], expected_lr(name), rel_tol=1e-12, abs_tol=1e-15)
                       for name in names), names[0]
        assert core.finite(core.schedule_state(opt, c, epoch))
    return observed


def run(freeze=True):
    tick = time.perf_counter()
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized(), 'CPU preflight must not initialize CUDA'
    c, sig = core.config_signature()
    preservation_before = protocol.verify()
    train, val, weights = core.development_data(c)
    assert len(train) == 8111 and len(val) == 902
    assert tuple(CLASSES) == tuple(c['class_order'])
    counts = train.label.value_counts().reindex(range(7)).to_numpy()
    expected_weights = np.sqrt(len(train) / (7. * counts))
    expected_weights /= expected_weights.mean()
    assert np.allclose(weights.numpy(), expected_weights, rtol=1e-6, atol=1e-7)
    assert np.allclose(c['loss_class_weights'], expected_weights, rtol=1e-12, atol=1e-12)
    outer_ids = {row['image_id'] for row in protocol.frozen_rows(0) if row['split'] == 'test'}
    assert not outer_ids & set(train.image_id) and not outer_ids & set(val.image_id)
    for wrong, training in ((val, True), (train, False), (pd.DataFrame([{'split': 'test'}]), False)):
        try:
            core.Images(wrong, c, training)
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong or assessment partition accepted')
    loader = core.epoch_loader(train, c, 3, True)
    order = list(iter(loader.sampler))
    assert len(order) == len(train) and set(order) == set(range(len(train)))
    assert order == list(iter(core.epoch_loader(train, c, 3, True).sampler))
    assert order != list(iter(core.epoch_loader(train, c, 4, True).sampler))
    batches = math.ceil(len(train) / c['batch_size'])
    groups = math.ceil(batches / c['accumulation_steps'])
    remainder = len(train) % c['effective_batch_size']
    assert c['batch_size'] in (4, 8) and c['batch_size'] * c['accumulation_steps'] == 32
    assert batches == math.ceil(8111 / c['batch_size']) and groups == 254 and remainder == 15
    assert len(loader) == batches and not loader.drop_last
    del loader, order
    accumulation_ok = weighted_accumulation_checks(c['batch_size'])
    counters_ok = meaningful_counter_checks(c)
    buffers_ok = ema_buffer_checks(c)
    resume_probes = synthetic_resume_checks(c, sig)
    random.seed(c['seed'])
    np.random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    dataset = core.Images(train, c, True)
    x, label, image_id = dataset[0]
    assert x.shape == (3, 224, 224) and core.finite(x)
    x, y = x.unsqueeze(0), torch.tensor([label])
    vdataset = core.Images(val, c, False)
    vx = vdataset[0][0]
    assert vx.shape == (3, 224, 224) and torch.equal(vx, vdataset[0][0])
    model = core.make_model(c, pretrained=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == c['parameter_count_with_seven_class_head'] == 87699975
    classifier = model.get_classifier()
    assert isinstance(classifier, nn.Linear) and classifier.in_features == 1024 and classifier.out_features == 7
    assert model.head.drop.p == c['head_dropout']
    opt = core.make_optimizer(model, c)
    schedule = optimizer_checks(model, opt, c)
    core.apply_learning_rates(opt, c, 0)
    initial = build_payload(c, sig, model, opt, None, 0, [], dict.fromkeys(WINNERS),
                            dict(raw=None, ema=None), 0, 0, [], 0., cuda_rng=False)
    checked_resume(initial, c, sig, require_cuda_rng=False)
    del initial
    core.set_stage(model, c, 1)
    head_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert head_names and all(name.startswith(tuple(c['warmup_trainable_prefixes'])) for name in head_names)
    assert not model.stem.training and not model.stages.training and model.head.training
    first_backbone = next(model.stem.parameters())
    head_before = classifier.weight.detach().clone()
    backbone_before = first_backbone.detach().clone()
    core.apply_learning_rates(opt, c, 1)
    logits = model(x)
    loss = F.cross_entropy(logits.float(), y, weight=weights)
    assert logits.shape == (1, 7) and core.finite([logits, loss])
    loss.backward()
    assert all(parameter.grad is not None and core.finite(parameter.grad)
               for parameter in model.parameters() if parameter.requires_grad)
    assert first_backbone.grad is None
    torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip_norm'], error_if_nonfinite=True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert not torch.equal(head_before, classifier.weight)
    assert torch.equal(backbone_before, first_backbone)
    core.set_stage(model, c, 3)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.stem.training and model.stages.training and model.head.training
    core.apply_learning_rates(opt, c, 3)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ema_head_before = ema.get_classifier().weight.detach().clone()
    logits = model(x)
    loss = F.cross_entropy(logits.float(), y, weight=weights)
    assert core.finite([logits, loss])
    loss.backward()
    assert all(parameter.grad is not None and core.finite(parameter.grad) for parameter in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip_norm'], error_if_nonfinite=True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert not torch.equal(backbone_before, first_backbone)
    core.update_ema(ema, model, c['ema']['decay'])
    expected_ema_head = ema_head_before.mul(c['ema']['decay']).add(classifier.weight.detach(), alpha=1.-c['ema']['decay'])
    assert torch.allclose(ema.get_classifier().weight, expected_ema_head, rtol=1e-6, atol=1e-8)
    model.eval()
    with torch.inference_mode():
        expected_raw = model(x).clone()
        expected_ema = ema(x).clone()
    assert core.finite([model.state_dict(), ema.state_dict(), opt.state_dict(), expected_raw, expected_ema])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='r202_cpu_preflight_', dir=OUTPUT) as temp:
        path = Path(temp).resolve()
        path.relative_to(OUTPUT.resolve())
        checkpoint = path / 'checkpoint.pt'
        state = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                     schedule=core.schedule_state(opt, c, 3), rng=core.capture_rng(False),
                     config=c, signature=sig)
        atomic_checkpoint(checkpoint, state)
        checkpoint_size = checkpoint.stat().st_size
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert loaded['config'] == c and loaded['signature'] == sig
        assert loaded['schedule'] == state['schedule']
        assert_state_equal(state['model'], loaded['model'])
        assert_state_equal(state['ema'], loaded['ema'])
        model.load_state_dict(loaded['model'], strict=True)
        ema.load_state_dict(loaded['ema'], strict=True)
        opt.load_state_dict(loaded['optimizer'])
        with torch.inference_mode():
            assert torch.equal(expected_raw, model(x))
            assert torch.equal(expected_ema, ema(x))
        core.restore_rng(loaded['rng'])
        values = (random.random(), np.random.random(), torch.rand(3))
        core.restore_rng(loaded['rng'])
        assert values[0] == random.random() and values[1] == np.random.random() and torch.equal(values[2], torch.rand(3))
        del loaded, state
    del model, ema, opt
    gc.collect()
    assert not torch.cuda.is_initialized()
    preservation_after = protocol.verify()
    result = dict(status='passed', signature=sig, cpu_only=True, gpu_used=False, cuda_initialized=False,
                  experiment_training_launched=False, model_accuracy_computed=False,
                  outer_assessment_images_opened=0, inner_training_samples_checked=1,
                  inner_validation_samples_checked=1, train_images=len(train), validation_images=len(val),
                  exposures_per_epoch=len(train), microbatches_per_epoch=batches,
                  accumulation_groups_per_epoch=groups, final_partial_group_images=remainder,
                  natural_sampling_reproducible=True, weighted_effective_batch_gradient_checked=accumulation_ok,
                  meaningful_counter_thresholds_and_fixed50_epoch_budget_checked=counters_ok,
                  model_parameters=count, fresh_pretrained_sha256=sha256(core.WEIGHT_FILE),
                  head_only_then_full_fine_tuning_checked=True, optimizer_coverage_and_no_decay_checked=True,
                  schedule_epoch_probes=schedule, finite_forward_backward_optimizer_steps=True,
                  cpu_optimizer_steps=2, ema_parameter_and_buffer_policy_checked=buffers_ok,
                  raw_ema_optimizer_schedule_rng_roundtrip=True, temporary_checkpoint_bytes=checkpoint_size,
                  native_initial_committed_schema_checked=True, resume_and_selector_probes=resume_probes,
                  wrong_partition_rejected=True, deterministic_identity_validation_checked=True,
                  historical_preservation_before=preservation_before, historical_preservation_after=preservation_after,
                  runtime=runtime_versions(), elapsed_seconds=time.perf_counter()-tick,
                  note='Two disposable CPU optimizer updates are correctness probes. Their states are discarded; no training experiment or accuracy is recorded.')
    write_json(OUTPUT / 'r202_convnext_base_preflight.json', result)
    if not freeze:
        return result
    if core.FREEZE.exists():
        existing = json.loads(core.FREEZE.read_text(encoding='utf-8'))
        if existing['signature'] != sig:
            raise ValueError('Existing launch freeze differs; preserve and inspect before amendment')
    else:
        write_json(core.FREEZE, dict(status='prepared_pending_gpu_approval', signature=sig,
                                   question=c['question'], experiment_id=c['experiment_id'],
                                   outer_assessment_enabled=False, preflight_cpu_only=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-freeze', action='store_true', help='Run correctness checks without writing a launch freeze')
    args = parser.parse_args()
    print(json.dumps(run(freeze=not args.no_freeze), indent=2))


if __name__ == '__main__':
    main()
