"""CPU correctness checks and launch freeze; never produce a model result."""
import argparse
import json
import random
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from research.common import ROOT, CLASSES, sha256, write_json
from research.train import atomic_checkpoint, runtime_versions
from . import core
from . import prepare_protocol as protocol

OUTPUT = ROOT/'results/image_level_replication/v2/preparation'


def policy_checks(c):
    def history(n, increasing=False):
        return [dict(epoch=e, val_accuracy=.85+(e*.0021 if increasing else (e%2)*.0004),
                     val_macro_f1=.8+(e*.0031 if increasing else (e%3)*.0004),
                     lr=.0001 if e<=10 else .00005, lr_after=.00005 if e>=10 else .0001)
                for e in range(1,n+1)]
    assert not core.stopping(history(24),c)['stop']
    flat = core.stopping(history(25),c)
    assert flat['stop'] and flat['stale']==24 and flat['meaningful_improvement_epochs']==[1]
    assert not core.stopping(history(40,True),c)['stop']
    assert core.stopping(history(50,True),c)['reason']=='maximum_epochs'
    no_lr = history(35)
    for row in no_lr:
        row['lr_after'] = row['lr'] = .0001
    assert not core.stopping(no_lr,c)['stop']
    # The second meaningful rise is measured from the previous meaningful anchor.
    accumulating = [dict(epoch=e,val_accuracy=.85+(e-1)*.0005,val_macro_f1=.8,
                         lr=.0001,lr_after=.0001) for e in range(1,7)]
    assert core.stopping(accumulating,c)['meaningful_improvement_epochs']==[1,5]
    late = history(32)
    for e in (24,):
        late[e-1]['val_accuracy'] = .855
    assert core.stopping(late,c)['reason']=='late_flat_trend_after_lr_reduction'


def run():
    tick = time.perf_counter()
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized(), 'CPU preflight must not initialize CUDA'
    c,sig = core.config_signature()
    protocol.verify()
    train,val = core.development_data(c)
    outer = {r['image_id'] for r in protocol.frozen_rows(0) if r['split']=='test'}
    assert not outer & set(train.image_id) and not outer & set(val.image_id)
    for wrong, training in [(val,True),(train,False)]:
        try:
            core.Images(wrong,c,training)
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong partition accepted')
    try:
        core.Images(__import__('pandas').DataFrame([dict(split='test')]),c)
    except ValueError:
        pass
    else:
        raise AssertionError('Assessment dataset accepted')
    indices = core.balanced_indices(train,42+1009)
    assert indices == core.balanced_indices(train,42+1009)
    assert indices != core.balanced_indices(train,42+2018)
    assert set(indices)==set(range(len(train)))
    exposure = Counter(train.label.to_numpy()[indices].tolist())
    assert set(exposure.values())=={5430} and len(indices)==38010
    policy_checks(c)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    dataset = core.Images(train,c,True)
    positions = [int(np.flatnonzero(train.label.to_numpy()==i)[0]) for i in range(4)]
    batch = [dataset[i] for i in positions]
    x = torch.stack([item[0] for item in batch])
    y = torch.tensor([item[1] for item in batch])
    assert x.shape==(4,3,112,112) and core.finite(x)
    vdataset = core.Images(val,c,False)
    vx = vdataset[0][0]
    assert torch.equal(vx,vdataset[0][0]) and vx.shape==(3,112,112)
    model = core.make_model(c,True)
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.classifier.parameters())
    assert total==sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = core.make_optimizer(model,c)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters()}
    scheduler = core.make_scheduler(opt,c)
    old_head = model.classifier[-1].weight.detach().clone()
    old_backbone = next(model.features.parameters()).detach().clone()
    model.train()
    z = model(x)
    loss = torch.nn.functional.cross_entropy(z.float(),y)
    assert z.shape==(4,7) and core.finite([z,loss])
    loss.backward()
    assert all(p.grad is not None and core.finite(p.grad) for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert not torch.equal(old_head,model.classifier[-1].weight)
    assert not torch.equal(old_backbone,next(model.features.parameters()))
    model.eval()
    with torch.inference_mode():
        expected = model(x).clone()
    assert core.finite([model.state_dict(), opt.state_dict(), expected])
    # Verify real state and optimizer serialization, not just model construction.
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cpu_preflight_',dir=OUTPUT) as tmp:
        temp = Path(tmp).resolve()
        temp.relative_to(OUTPUT.resolve())
        file = temp/'checkpoint.pt'
        state = dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                     rng=core.capture_rng(False),config=c,signature=sig)
        atomic_checkpoint(file,state)
        loaded = torch.load(file,map_location='cpu',weights_only=False)
        model.load_state_dict(loaded['model'],strict=True)
        opt.load_state_dict(loaded['optimizer'])
        scheduler.load_state_dict(loaded['scheduler'])
        with torch.inference_mode():
            assert torch.equal(expected,model(x))
        core.restore_rng(loaded['rng'])
        values = (random.random(),np.random.random(),torch.rand(3))
        core.restore_rng(loaded['rng'])
        assert values[0]==random.random() and values[1]==np.random.random() and torch.equal(values[2],torch.rand(3))
    assert not torch.cuda.is_initialized()
    protocol.verify()
    result = dict(status='passed', cpu_only=True, cuda_initialized=False, gpu_used=False,
                  experiment_training_launched=False, model_accuracy_computed=False,
                  outer_assessment_images_opened=0, inner_training_samples_checked=4,
                  inner_validation_samples_checked=1, train_images=len(train),validation_images=len(val),
                  epoch_training_exposures=len(indices),exposures_per_class=5430,
                  train_batches=math_ceil(len(indices)/c['batch_size']),
                  model_parameters=total,head_parameters=head,backbone_parameters=total-head,
                  fresh_pretrained_sha256=sha256(core.WEIGHT_FILE),
                  all_parameters_trainable=True,finite_forward_backward_optimizer_step=True,
                  wrong_partition_rejected=True,balanced_exposure_reproducible=True,
                  meaningful_plateau_policy_checked=True,checkpoint_optimizer_rng_roundtrip=True,
                  historical_preservation_checked=True,runtime=runtime_versions(),
                  elapsed_seconds=time.perf_counter()-tick,
                  note='One temporary CPU optimizer step is a correctness probe; its state is discarded. No run or result is created.')
    write_json(OUTPUT/'b3_fold00_preflight.json',result)
    if core.FREEZE.exists():
        if json.loads(core.FREEZE.read_text())['signature'] != sig:
            raise ValueError('Existing launch freeze differs; preserve and inspect before amendment')
    else:
        write_json(core.FREEZE,dict(status='prepared_pending_gpu_approval',signature=sig,
                  question=c['question'],experiment_id=c['experiment_id'],outer_assessment_enabled=False))
    return result


def math_ceil(value):
    import math
    return math.ceil(value)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(),indent=2))


if __name__=='__main__':
    main()
