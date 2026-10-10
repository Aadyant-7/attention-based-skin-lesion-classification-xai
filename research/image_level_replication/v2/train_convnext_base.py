"""R202: isolated 50-epoch ConvNeXt-V2 Base, raw/EMA inner validation.

No outer-assessment loader exists. --check/--closeout never initialize CUDA.
Resume replays an interrupted partial epoch from the atomic committed state.
"""
import argparse
import copy
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from research.common import ROOT, CLASSES, relative, sha256, write_json, write_csv
from research.plots import metric_figures, training_figures, comparison_figures
from research.registry import FIELDS
from .convnext_core import (CONFIG, FREEZE, config_signature, development_data, epoch_loader,
                            make_model, make_optimizer, set_stage, apply_learning_rates,
                            schedule_state, update_ema, meaningful, finite, capture_rng,
                            restore_rng, report)
from .train_b3 import (atomic_checkpoint, run_lock, model_cpu, state_cpu,
                      runtime_versions as base_versions, validate_prediction_record,
                      write_prediction_package, update_registries, PROBABILITY_COLUMNS)

WINNERS = ('raw_accuracy', 'raw_macro_f1', 'ema_accuracy', 'ema_macro_f1')
METRICS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')


def runtime_versions():
    return dict(base_versions(), timm=version('timm'))


def require_launch_freeze(c, signature):
    if not FREEZE.is_file() or json.loads(FREEZE.read_text(encoding='utf-8'))['signature'] != signature:
        raise ValueError('R202 config/code/weights changed or CPU launch freeze is absent; rerun preparation')


def selected(winners, metric='accuracy'):
    values = [winners[branch + '_' + metric] for branch in ('raw', 'ema') if winners[branch + '_' + metric]]
    return max(values, key=lambda r: (r['metrics'][metric], -r['epoch'], r['branch'] == 'raw')) if values else None


def select_winners(winners, records, models):
    for branch in ('raw', 'ema'):
        record = records.get(branch)
        if record is None:
            continue
        improved = [metric for metric in ('accuracy', 'macro_f1')
                    if winners[branch + '_' + metric] is None or
                    record['metrics'][metric] > winners[branch + '_' + metric]['metrics'][metric]]
        if improved:
            snapshot = dict(record, model=model_cpu(models[branch]))
            for metric in improved:
                winners[branch + '_' + metric] = snapshot
    return winners


def build_payload(c, signature, model, optimizer, ema, epoch, history, winners,
                  last, updates, ema_updates, events, runtime_seconds,
                  status='running', cuda_rng=True, consecutive=0):
    state = dict(config=c, signature=signature, epoch=epoch, model=model_cpu(model),
                 ema_model=model_cpu(ema) if ema is not None else None,
                 optimizer=state_cpu(optimizer.state_dict()),
                 lr_schedule_state=schedule_state(optimizer, c, epoch),
                 rng=capture_rng(cuda_rng), history=copy.deepcopy(history), winners=winners,
                 last_validation=last, optimizer_updates=updates, ema_updates=ema_updates,
                 consecutive_nonfinite_updates=consecutive,
                 numerical_events=copy.deepcopy(events), meaningful_stopping=meaningful(history, c),
                 runtime_versions=runtime_versions(), runtime_seconds=runtime_seconds,
                 status=status, class_order=list(CLASSES), inference_precision='fp32',
                 outer_assessment_loaded=False,
                 checkpoint_semantics='Complete epoch; interrupted partial epoch replays from here')
    if not finite(state):
        raise FloatingPointError('Refusing nonfinite R202 committed state')
    return state


def checked_resume(saved, c, signature, require_cuda_rng=True):
    if saved.get('config') != c or saved.get('signature') != signature or saved.get('runtime_versions') != runtime_versions():
        raise ValueError('R202 committed config/code/weights/runtime differ')
    if saved.get('class_order') != list(CLASSES) or saved.get('inference_precision') != 'fp32' or saved.get('outer_assessment_loaded') is not False:
        raise ValueError('R202 class order/precision/partition mismatch')
    e, history = saved.get('epoch'), saved.get('history')
    if not isinstance(e, int) or not 0 <= e <= 50 or not isinstance(history, list) or [r['epoch'] for r in history] != list(range(1, e + 1)):
        raise ValueError('R202 history is not contiguous through the committed epoch')
    if not finite(saved):
        raise FloatingPointError('Nonfinite R202 checkpoint field')
    if require_cuda_rng and not saved.get('rng', {}).get('cuda'):
        raise ValueError('Committed checkpoint lacks CUDA RNG')
    if set(saved.get('winners', {})) != set(WINNERS):
        raise ValueError('R202 raw/EMA selectors missing')
    for key in ('optimizer_updates', 'ema_updates', 'consecutive_nonfinite_updates'):
        if type(saved.get(key)) is not int or saved[key] < 0:
            raise ValueError('Invalid accepted-update counter: ' + key)
    if any(type(r.get('accepted_updates')) is not int or r['accepted_updates'] <= 0 for r in history):
        raise ValueError('History has invalid accepted-update counts')
    if saved['consecutive_nonfinite_updates'] >= c['max_consecutive_nonfinite_updates'] or saved['consecutive_nonfinite_updates'] != (history[-1].get('consecutive_nonfinite_updates', 0) if history else 0):
        raise ValueError('Consecutive nonfinite-update counter disagrees with committed history')
    if saved['optimizer_updates'] != sum(r['accepted_updates'] for r in history):
        raise ValueError('Optimizer update count disagrees with history')
    if saved['ema_updates'] != sum(r['accepted_updates'] for r in history if r['epoch'] > 2):
        raise ValueError('EMA update count disagrees with accepted full-fine-tuning steps')
    if (saved['ema_model'] is None) != (e <= 2):
        raise ValueError('EMA availability disagrees with the fixed warmup')
    if saved['meaningful_stopping'] != meaningful(history, c):
        raise ValueError('Meaningful counters disagree with history')
    groups = saved['optimizer']['param_groups']
    pseudo = type('SavedOptimizer', (), {'param_groups': groups})()
    if saved['lr_schedule_state'] != schedule_state(pseudo, c, e):
        raise ValueError('Saved learning-rate schedule disagrees with epoch')
    for branch in ('raw', 'ema'):
        available = [r for r in history if r.get(branch + '_val_accuracy') is not None]
        for metric in ('accuracy', 'macro_f1'):
            winner = saved['winners'][branch + '_' + metric]
            if not available:
                if winner is not None:
                    raise ValueError('Selector exists before branch validation')
                continue
            expected = max(available, key=lambda r: r[branch + '_val_' + metric])
            if winner is None or winner['branch'] != branch or winner['epoch'] != expected['epoch'] or winner['metrics'][metric] != expected[branch + '_val_' + metric]:
                raise ValueError('Raw/EMA best selector disagrees with earliest history maximum')
        last = saved['last_validation'].get(branch)
        if (last is None) != (not available) or (last is not None and last['epoch'] != e):
            raise ValueError('Final branch validation does not match the committed epoch')


def write_package(record, folder, val, c, figures=False):
    write_prediction_package(record, folder, val, c, figures=False)
    path = folder / 'validation_metrics.json'
    metrics = json.loads(path.read_text(encoding='utf-8'))
    metrics.update(model=c['model_name'], branch=record['branch'])
    write_json(path, metrics)
    if figures:
        metric_figures(metrics, folder / 'figures',
                       'ConvNeXt-V2 Base | ' + record['branch'].upper() + ' | inner validation epoch ' + str(record['epoch']))


def checkpoint_record(record, c, signature, kind):
    return dict(record, config=c, signature=signature, checkpoint_kind=kind,
                class_order=list(CLASSES), inference_precision='fp32', outer_assessment_loaded=False,
                selection_partition='inner_validation_fold00')


def registry_row(saved, c, status):
    best = selected(saved['winners'])
    row = {key: '' for key in FIELDS}
    row.update(experiment_id=c['experiment_id'], era='structured', record_kind='training_run',
               phase='image_level_replication_v2', protocol=c['protocol'], evaluation_split='inner_validation_fold00',
               split_manifest=c['split_manifest'], split_sha256=c['split_sha256'],
               method='Fresh ConvNeXt-V2 Base; stage-wise AdamW/cosine; fixed raw-versus-EMA selection',
               model=c['model_name'], pretrained_weights=c['weights'], attention=c['attention'],
               image_size=c['image_size'], preprocessing=c['preprocessing'], augmentation=c['augmentation'],
               imbalance=c['imbalance'], loss=c['loss'], optimizer=c['optimizer'], backbone_lr=c['backbone_lr'],
               head_lr=c['head_lr'], batch_size=c['batch_size'], seed=c['seed'], epochs=saved['epoch'],
               runtime_seconds=saved['runtime_seconds'], config_path=c['config_path'],
               history_path=c['results'] + '/history.csv', status=status, decision='inner_validation_only',
               notes='Fresh author external weights; V2 split with natural lesion overlap. No CBAM, outer inference or legacy checkpoint fusion. '
                     '50 planned epochs; meaningful counters observational; manual stop honored at committed epoch. '
                     'Two predeclared branches, raw and fixed EMA0.999; earliest maximum accuracy, raw at same-epoch ties.')
    if best:
        row.update(best_epoch=best['epoch'], **{k: best['metrics'][k] for k in METRICS},
                   val_loss=best['metrics']['loss'], checkpoint=c['checkpoints'] + '/best.pt',
                   checkpoint_available_local=True, metrics_path=c['results'] + '/best_accuracy/validation_metrics.json',
                   confusion_matrix_path=c['results'] + '/best_accuracy/confusion_matrix.csv',
                   plots_dir=c['results'] + '/best_accuracy/figures', source_selector=best['branch'])
        if (ROOT / row['checkpoint']).is_file():
            row['checkpoint_sha256'] = sha256(ROOT / row['checkpoint'])
    return row


def repair(saved, folder, ck, c, val, status='running', force=False):
    if saved['history']:
        write_csv(folder / 'history.csv', saved['history'])
    write_csv(folder / 'lr_history.csv', [dict(epoch=r['epoch'], **r['group_lrs']) for r in saved['history']])
    write_json(folder / 'meaningful_stopping.json', saved['meaningful_stopping'])
    write_json(folder / 'numerical_events.json', saved['numerical_events'])
    write_json(folder / 'progress.json', dict(status=status, pid=os.getpid(), committed_epoch=saved['epoch'],
               max_epochs=50, optimizer_updates=saved['optimizer_updates'], ema_updates=saved['ema_updates'],
               runtime_seconds=saved['runtime_seconds'], meaningful_stopping=saved['meaningful_stopping'],
               results=c['results'], checkpoints=c['checkpoints'], outer_assessment_loaded=False,
               stopping_reason=saved.get('stop_reason', 'maximum_epochs' if saved['epoch'] == 50 else 'running')))
    for key, record in saved['winners'].items():
        if record:
            target = ck / ('best_' + key + '.pt')
            if force or record['epoch'] == saved['epoch'] or not target.is_file():
                atomic_checkpoint(target, checkpoint_record(record, c, saved['signature'], key))
            write_package(record, folder / key, val, c)
    for metric, name, destination in (('accuracy', 'best.pt', 'best_accuracy'),
                                      ('macro_f1', 'best_macro_f1.pt', 'best_macro_f1')):
        record = selected(saved['winners'], metric)
        if record:
            if force or record['epoch'] == saved['epoch'] or not (ck / name).is_file():
                atomic_checkpoint(ck / name, checkpoint_record(record, c, saved['signature'], metric))
            write_package(record, folder / destination, val, c)
            if metric == 'accuracy':
                write_package(record, folder, val, c)
    for branch, record in saved['last_validation'].items():
        if record:
            write_package(record, folder / ('final_' + branch), val, c)
    row = registry_row(saved, c, status)
    update_registries(row)
    write_json(folder / 'record.json', row)


def closeout(saved, folder, ck, c, val):
    if saved['epoch'] == 0:
        raise ValueError('No completed validation epoch to close out')
    if saved['epoch'] < 50 and saved.get('status') != 'stopped_by_user':
        raise ValueError('Unfinished run; explicit resume or epoch-boundary stop required')
    status = 'completed' if saved['epoch'] == 50 else 'stopped_by_user'
    repair(saved, folder, ck, c, val, status=status)
    final = dict(config=c, signature=saved['signature'], epoch=saved['epoch'],
                 model=saved['model'], ema_model=saved['ema_model'],
                 final_validation=saved['last_validation'], status=status,
                 class_order=list(CLASSES), inference_precision='fp32', outer_assessment_loaded=False,
                 checkpoint_kind='final_raw_and_ema_weights; resume from latest.pt')
    final_path = ck / 'final.pt'
    if final_path.is_file():
        previous = torch.load(final_path, map_location='cpu', weights_only=False)
        if previous['epoch'] != saved['epoch']:
            backup = ck / ('final_epoch_' + str(previous['epoch']) + '_' + str(time.time_ns()) + '.pt')
            final_path.rename(backup)
    atomic_checkpoint(final_path, final)
    records = dict(saved['winners'], best_accuracy=selected(saved['winners']),
                   best_macro_f1=selected(saved['winners'], 'macro_f1'))
    records.update({'final_' + branch: record for branch, record in saved['last_validation'].items()})
    for destination, record in records.items():
        if record:
            write_package(record, folder / destination, val, c, figures=True)
    frame = pd.DataFrame(saved['history'])
    training_figures(frame, folder / 'figures', 'ConvNeXt-V2 Base | raw | inner validation', selection_metric='accuracy')
    from research.plots import plt, save
    fig, ax = plt.subplots(figsize=(7, 4))
    for group_name in frame.iloc[0]['group_lrs']:
        ax.plot(frame.epoch, [groups[group_name] for groups in frame.group_lrs], label=group_name)
    ax.set(xlabel='Epoch', ylabel='Learning rate', title='ConvNeXt-V2 Base | declared stage-wise schedule')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.2)
    save(fig, folder / 'figures', 'learning_rate_curves')
    if saved['ema_model'] is not None:
        ema_frame = frame.copy()
        for key in ('loss', *METRICS):
            ema_frame['val_' + key] = frame['ema_val_' + key]
        training_figures(ema_frame, folder / 'figures/ema', 'Raw training / EMA inner validation', selection_metric='accuracy')
    rows = [dict(display_name=branch.upper() + ' accuracy-selected',
                 accuracy=saved['winners'][branch + '_accuracy']['metrics']['accuracy'],
                 macro_f1=saved['winners'][branch + '_accuracy']['metrics']['macro_f1'])
            for branch in ('raw', 'ema') if saved['winners'][branch + '_accuracy']]
    comparison_figures(rows, folder / 'comparison_figures', 'Fixed RAW / EMA comparison — V2 inner validation')
    checkpoint_files = [path for path in ck.glob('*.pt') if not path.name.startswith('failure_')]
    write_json(folder / 'closeout.json', dict(status=status, committed_epochs=saved['epoch'],
               stopping_reason=saved.get('stop_reason', 'maximum_epochs'),
               best_accuracy={k: v for k, v in selected(saved['winners']).items() if k != 'model'},
               best_macro_f1={k: v for k, v in selected(saved['winners'], 'macro_f1').items() if k != 'model'},
               final_validation=saved['last_validation'], meaningful_stopping=saved['meaningful_stopping'],
               optimizer_updates=saved['optimizer_updates'], ema_updates=saved['ema_updates'],
               numerical_events=saved['numerical_events'], outer_assessment_loaded=False,
               checkpoints={p.name: dict(path=relative(p), sha256=sha256(p)) for p in checkpoint_files},
               evaluation_claim='Single-fold inner validation; no outer/test or completed K10 result'))


def validation(model, loader):
    model.eval()
    ys, ids, arrays = [], [], []
    total_loss = 0.0
    with torch.inference_mode():
        for images, targets, image_ids in loader:
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            logits = model(images).float()  # No autocast in either validation branch.
            probabilities = logits.softmax(1)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            if not finite((images, logits, probabilities, loss)):
                raise FloatingPointError('Nonfinite FP32 validation; IDs=' + repr(list(image_ids)))
            total_loss += float(loss)
            ys.extend(targets.cpu().tolist())
            ids.extend(image_ids)
            arrays.append(probabilities.cpu().numpy())
    p = np.concatenate(arrays)
    return report(np.asarray(ys), p, total_loss / len(ys)), p, ids


def prediction_record(branch, epoch, model, loader, val):
    metrics, probabilities, ids = validation(model, loader)
    if ids != val.image_id.tolist():
        raise ValueError('Validation IDs/order differ from frozen inner validation')
    frame = val[['image_id', 'lesion_id']].copy()
    frame['true_class'] = val.diagnosis
    frame['predicted_class'] = [CLASSES[i] for i in probabilities.argmax(1)]
    frame[PROBABILITY_COLUMNS] = probabilities
    return dict(branch=branch, epoch=epoch, metrics=metrics, predictions=frame.to_dict('records'))


def effective_batches(loader, accumulation):
    group = []
    for batch in loader:
        group.append(batch)
        if len(group) == accumulation:
            yield group
            group = []
    if group:
        yield group


def stop_request(folder):
    path = folder / 'stop_request.json'
    if not path.is_file():
        return None
    request = json.loads(path.read_text(encoding='utf-8'))
    if request.get('experiment_id') != folder.name or not request.get('reason'):
        raise ValueError('Malformed explicit epoch-boundary stop request')
    return request


def commit_manual_stop(latest, folder, ck, c, val, request):
    """Honor a request received between epochs without an additional GPU update."""
    saved = torch.load(latest, map_location='cpu', weights_only=False)
    saved.update(status='stopped_by_user', stop_reason=request['reason'], stop_request=request)
    atomic_checkpoint(latest, saved)
    if saved['epoch']:
        closeout(saved, folder, ck, c, val)
    else:
        repair(saved, folder, ck, c, val, status='stopped_by_user')


def configure_cuda(c):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG', c['cuda_workspace_config']) != c['cuda_workspace_config']:
        raise ValueError('CUBLAS setting differs from the frozen deterministic config')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = c['cuda_workspace_config']
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Frozen run requires CUDA/BF16; no silent precision substitution')
    torch.set_num_threads(4)
    random.seed(c['seed'])
    np.random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)


def memory_check():
    """Explicit future GPU check; disposable synthetic updates, no experiment results."""
    c, signature = config_signature()
    require_launch_freeze(c, signature)
    result_path = FREEZE.parent / 'r202_convnext_base_gpu_feasibility.json'
    if result_path.exists():
        previous = result_path.with_name(result_path.stem + '_' + str(time.time_ns()) + '.json')
        result_path.rename(previous)
    result = dict(signature=signature, runtime=runtime_versions(), status='running',
                  synthetic_inputs_only=True, experiment_training_launched=False,
                  model_accuracy_computed=False, assessment_images_opened=0,
                  batch_size=c['batch_size'], accumulation_steps=c['accumulation_steps'],
                  validation_batch_size=c['validation_batch_size'])
    try:
        configure_cuda(c)
        result['device'] = torch.cuda.get_device_name()
        result['total_vram_bytes'] = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.reset_peak_memory_stats()
        model = make_model(c, pretrained=True).cuda()
        optimizer = make_optimizer(model, c)
        set_stage(model, c, 3)
        apply_learning_rates(optimizer, c, 3)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        weights = torch.tensor(c['loss_class_weights'], device='cuda', dtype=torch.float32)
        batch = c['batch_size']
        targets = torch.arange(batch, device='cuda') % 7
        denominator = weights[targets].sum() * c['accumulation_steps']
        for _ in range(c['accumulation_steps']):
            images = torch.randn(batch, 3, 224, 224, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images)
            loss = F.cross_entropy(logits.float(), targets, weight=weights, reduction='sum') / denominator
            if not finite((logits, loss)):
                raise FloatingPointError('Synthetic GPU feasibility logits/loss are nonfinite')
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()  # Materialize full AdamW state; this disposable state is never a checkpoint.
        update_ema(ema, model, c['ema']['decay'])
        torch.cuda.synchronize()
        result['full_stage_training_peak_bytes'] = torch.cuda.max_memory_allocated()
        optimizer.zero_grad(set_to_none=True)
        del images, logits, loss
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            images = torch.randn(c['validation_batch_size'], 3, 224, 224, device='cuda')
            for branch in (model.eval(), ema):
                logits = branch(images).float()  # FP32, no autocast, same as real identity validation.
                if not finite((logits, logits.softmax(1))):
                    raise FloatingPointError('Synthetic FP32 feasibility inference is nonfinite')
        torch.cuda.synchronize()
        result.update(status='passed', fp32_validation_peak_bytes=torch.cuda.max_memory_allocated(),
                      gpu_feasibility_measured=True, raw_and_resident_ema_checked=True)
    except BaseException as exc:
        result.update(status='failed_preserved', error=repr(exc), automatic_recipe_change=False)
        write_json(result_path, result)
        raise
    write_json(result_path, result)
    print(json.dumps(result, indent=2))


def require_memory_check(signature):
    path = FREEZE.parent / 'r202_convnext_base_gpu_feasibility.json'
    if not path.is_file():
        raise ValueError('After GPU approval, run --memory-check before --train; Phase 1 is CPU-only')
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('signature') != signature or result.get('runtime') != runtime_versions() or result.get('status') != 'passed':
        raise ValueError('GPU feasibility missing, failed, or from a different frozen setup')


def run(resume=False, allow_user_stop_resume=False):
    c, signature = config_signature()
    require_launch_freeze(c, signature)
    train, val, class_weights = development_data(c)
    folder, ck = ROOT / c['results'], ROOT / c['checkpoints']
    latest = ck / 'latest.pt'
    saved = None
    if resume:
        if not latest.is_file():
            raise FileNotFoundError('No committed latest.pt; inspect preserved failure evidence')
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        checked_resume(saved, c, signature)
        for record in list(saved['winners'].values()) + list(saved['last_validation'].values()):
            if record:
                validate_prediction_record(record, val)
        if saved['epoch'] == c['max_epochs']:
            closeout(saved, folder, ck, c, val)
            print('Completed run repaired from stored predictions; no CUDA initialization')
            return
        request = stop_request(folder)
        if (saved.get('status') == 'stopped_by_user' or request) and not allow_user_stop_resume:
            if request:
                commit_manual_stop(latest, folder, ck, c, val, request)
            elif saved['epoch']:
                closeout(saved, folder, ck, c, val)
            print('Explicit user stop preserved; --allow-user-stop-resume requires a new instruction')
            return
        if request:
            destination = folder / ('stop_request_honored_' + str(time.time_ns()) + '.json')
            (folder / 'stop_request.json').rename(destination)
    elif folder.exists() or ck.exists():
        raise FileExistsError('R202 already exists; use --resume, never --train again')
    require_memory_check(signature)
    required_free = max(4 * 1024 ** 3, latest.stat().st_size + 1024 ** 3) if resume else 10 * 1024 ** 3
    if shutil.disk_usage(ROOT).free < required_free:
        raise RuntimeError('Insufficient disk for atomic checkpoint replacement; need ' + str(required_free) + ' free bytes')
    if not resume:
        folder.mkdir(parents=True)
        ck.mkdir(parents=True)
        write_json(folder / 'config.json', c)
        write_json(folder / 'launch_manifest.json', signature)
        write_json(folder / 'environment.json', dict(runtime=runtime_versions(),
                   git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    logger = logging.getLogger(c['experiment_id'])
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(folder / 'train.log', encoding='utf-8'), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    tick = time.perf_counter()
    base_runtime = (saved or {}).get('runtime_seconds', 0)
    epoch = (saved or {}).get('epoch', 0)
    history = (saved or {}).get('history', [])
    winners = (saved or {}).get('winners', {name: None for name in WINNERS})
    last = (saved or {}).get('last_validation', dict(raw=None, ema=None))
    updates = (saved or {}).get('optimizer_updates', 0)
    ema_updates = (saved or {}).get('ema_updates', 0)
    events = (saved or {}).get('numerical_events', [])
    consecutive = (saved or {}).get('consecutive_nonfinite_updates', 0)
    try:
        configure_cuda(c)
        model = make_model(c, pretrained=not resume).cuda()
        optimizer = make_optimizer(model, c)
        ema = None
        if saved:
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            if saved['ema_model'] is not None:
                ema = copy.deepcopy(model).eval().requires_grad_(False)
                ema.load_state_dict(saved['ema_model'], strict=True)
            repair(saved, folder, ck, c, val, force=True)
            restore_rng(saved['rng'])
        else:
            set_stage(model, c, 0)
            apply_learning_rates(optimizer, c, 0)
            saved = build_payload(c, signature, model, optimizer, ema, 0, history,
                                  winners, last, updates, ema_updates, events, 0)
            atomic_checkpoint(latest, saved)
            repair(saved, folder, ck, c, val)
        del saved
        class_weights = class_weights.cuda()
        write_json(folder / 'process.json', dict(pid=os.getpid(), resume=resume, committed_epoch=epoch,
                   results=c['results'], checkpoints=c['checkpoints'], outer_assessment_loaded=False))
        logger.info('LOGGING/CHECKPOINTING ACTIVE resume=%s committed_epoch=%d train=%d inner_val=%d latest=%s; no outer/test loader',
                    resume, epoch, len(train), len(val), relative(latest))
        first_accepted = False
        for epoch in range(epoch + 1, c['max_epochs'] + 1):
            request = stop_request(folder)
            if request:
                commit_manual_stop(latest, folder, ck, c, val, request)
                logger.info('USER STOP honored before epoch %d; previous boundary preserved', epoch)
                return
            epoch_tick = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            stage = set_stage(model, c, epoch)
            group_lrs = apply_learning_rates(optimizer, c, epoch)
            if epoch == 3 and ema is None:
                ema = copy.deepcopy(model).eval().requires_grad_(False)
                logger.info('EMA initialized after committed head warmup; decay=%g', c['ema']['decay'])
            loader = epoch_loader(train, c, epoch, True)
            groups = (len(loader) + c['accumulation_steps'] - 1) // c['accumulation_steps']
            numerator = mass = 0.0
            correct = accepted = seen = skipped = accepted_updates = 0
            for step, microbatches in enumerate(effective_batches(loader, c['accumulation_steps']), 1):
                ids = [image_id for _, _, image_ids in microbatches for image_id in image_ids]
                diagnostic_batch = dict(epoch=epoch, effective_step=step, image_ids=ids)
                seen += len(ids)
                denominator = class_weights[torch.cat([targets for _, targets, _ in microbatches]).cuda()].sum()
                optimizer.zero_grad(set_to_none=True)
                buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
                group_numerator = 0.0
                group_correct = 0
                for images, targets, image_ids in microbatches:
                    images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                    if not torch.isfinite(images).all():
                        raise FloatingPointError('Nonfinite training input; IDs=' + repr(list(image_ids)))
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(images)
                    loss_sum = F.cross_entropy(logits.float(), targets, weight=class_weights, reduction='sum')
                    if not finite((logits, loss_sum)):
                        raise FloatingPointError('Nonfinite BF16 logits/FP32 training loss; IDs=' + repr(list(image_ids)))
                    (loss_sum / denominator).backward()
                    group_numerator += float(loss_sum.detach())
                    group_correct += int((logits.detach().argmax(1) == targets).sum())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip_norm'], error_if_nonfinite=False)
                if not torch.isfinite(norm):
                    bad = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                    with torch.no_grad():
                        for name, buffer in model.named_buffers():
                            buffer.copy_(buffers[name])
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    consecutive += 1
                    events.append(dict(diagnostic_batch, kind='nonfinite_gradient_effective_update_skipped',
                                       nonfinite_parameters=bad, consecutive=consecutive,
                                       ema_updated=False, buffers_restored=True))
                    write_json(folder / 'numerical_events_live.json', events)
                    logger.warning('NONFINITE gradient; whole effective group skipped epoch=%d step=%d consecutive=%d', epoch, step, consecutive)
                    if consecutive >= c['max_consecutive_nonfinite_updates']:
                        raise FloatingPointError('Consecutive nonfinite update limit; committed latest preserved')
                    continue
                optimizer.step()
                updates += 1
                accepted_updates += 1
                consecutive = 0
                if ema is not None:
                    update_ema(ema, model, c['ema']['decay'])
                    ema_updates += 1
                numerator += group_numerator
                mass += float(denominator)
                correct += group_correct
                accepted += len(ids)
                if not first_accepted:
                    first_accepted = True
                    write_json(folder / 'first_accepted_update.json', dict(epoch=epoch, effective_step=step,
                               optimizer_updates=updates, committed_checkpoint=relative(latest),
                               checkpoint_exists=latest.is_file(), log_active=True))
                    logger.info('FIRST ACCEPTED GPU UPDATE; logging and committed recovery checkpoint active')
                if step == 1 or step % 25 == 0 or step == groups:
                    logger.info('epoch %d/50 effective update %d/%d stage=%s weighted_CE=%.6f',
                                epoch, step, groups, stage, group_numerator / float(denominator))
            if seen != len(train) or accepted == 0:
                raise FloatingPointError('Natural sampling coverage or accepted updates invalid')
            optimizer.zero_grad(set_to_none=True)  # Match the checked FP32-validation memory footprint.
            last = dict(raw=prediction_record('raw', epoch, model, epoch_loader(val, c, epoch, False), val),
                        ema=prediction_record('ema', epoch, ema, epoch_loader(val, c, epoch, False), val) if ema is not None else None)
            raw = last['raw']['metrics']
            row = dict(epoch=epoch, stage=stage, epoch_seed=c['seed'] + epoch * 1009,
                       train_loss=numerator / mass, train_accuracy=correct / accepted,
                       train_images_seen=seen, train_images_accepted=accepted,
                       accepted_updates=accepted_updates, optimizer_updates=updates, ema_updates=ema_updates,
                       consecutive_nonfinite_updates=consecutive,
                       nonfinite_updates_skipped=skipped, group_lrs=group_lrs,
                       lr=group_lrs['stage3_decay'], lr_after=group_lrs['stage3_decay'],
                       training_precision=c['training_precision'], validation_precision='fp32',
                       epoch_seconds=time.perf_counter() - epoch_tick,
                       peak_allocated_vram_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)
            for key in ('loss', *METRICS):
                row['val_' + key] = raw[key]
                row['raw_val_' + key] = raw[key]
                row['ema_val_' + key] = last['ema']['metrics'][key] if last['ema'] else None
            history.append(row)
            policy = meaningful(history, c)
            row.update(meaningful_stale=policy['stale'], last_meaningful_epoch=policy['last_meaningful_epoch'])
            select_winners(winners, last, dict(raw=model, ema=ema))
            request = stop_request(folder)
            status = 'completed' if epoch == 50 else ('stopped_by_user' if request else 'running')
            committed = build_payload(c, signature, model, optimizer, ema, epoch, history, winners,
                                      last, updates, ema_updates, events,
                                      base_runtime + time.perf_counter() - tick, status=status, consecutive=consecutive)
            if request:
                committed.update(stop_reason=request['reason'], stop_request=request)
            atomic_checkpoint(latest, committed)  # Epoch commit precedes CSV/registry/bookkeeping.
            repair(committed, folder, ck, c, val, status=status)
            logger.info('EPOCH %d raw_accuracy=%.6f raw_macroF1=%.6f EMA_accuracy=%s EMA_macroF1=%s best_epoch=%d meaningful_stale=%d status=%s',
                        epoch, raw['accuracy'], raw['macro_f1'], row['ema_val_accuracy'], row['ema_val_macro_f1'],
                        selected(winners)['epoch'], policy['stale'], status)
            del committed, loader
            if status != 'running':
                break
        committed = torch.load(latest, map_location='cpu', weights_only=False)
        checked_resume(committed, c, signature)
        closeout(committed, folder, ck, c, val)
        logger.info('CLOSED at epoch=%d status=%s; stored inner predictions only; no outer/test inference or next experiment',
                    committed['epoch'], committed['status'])
    except BaseException as exc:
        stamp = 'failure_' + str(time.time_ns())
        committed = None
        secondary_errors = []
        if latest.is_file():
            try:
                committed = torch.load(latest, map_location='cpu', weights_only=False)
            except Exception as error:
                secondary_errors.append('Reading latest: ' + repr(error))
        info = dict(error=repr(exc), attempted_epoch=epoch,
                    latest_committed_epoch=committed['epoch'] if committed else None,
                    latest_valid_checkpoint=relative(latest) if committed else None,
                    batch=locals().get('diagnostic_batch'), numerical_events=events,
                    automatic_retry=False, outer_assessment_loaded=False, secondary_errors=secondary_errors)
        write_json(folder / (stamp + '.json'), info)
        if 'model' in locals():
            try:
                atomic_checkpoint(ck / (stamp + '_diagnostic.pt'), dict(diagnostic_only=True,
                                  attempted_epoch=epoch, config=c, signature=signature,
                                  model=model_cpu(model), batch=locals().get('diagnostic_batch'),
                                  numerical_events=events, inference_or_resume_eligible=False))
            except Exception as error:
                secondary_errors.append('Saving diagnostic weights: ' + repr(error))
        if committed:
            try:
                repair(committed, folder, ck, c, val, status='failed_preserved')
            except Exception as error:
                secondary_errors.append('Failure bookkeeping: ' + repr(error))
        write_json(folder / (stamp + '.json'), info)
        write_json(folder / 'progress.json', dict(status='failed_preserved', **info))
        logger.exception('STOPPED; latest committed checkpoint and failure evidence preserved')
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--closeout', action='store_true')
    mode.add_argument('--request-stop', action='store_true')
    mode.add_argument('--memory-check', action='store_true', help='Requires GPU approval; disposable synthetic feasibility check')
    parser.add_argument('--reason', default='Explicit user request to stop at a completed epoch')
    parser.add_argument('--allow-user-stop-resume', action='store_true')
    args = parser.parse_args()
    if args.allow_user_stop_resume and not args.resume:
        parser.error('--allow-user-stop-resume is only valid with --resume')
    c, signature = config_signature()
    if args.check:
        train, val, _ = development_data(c)
        print(json.dumps(dict(status='prepared_pending_gpu_approval', train=len(train), inner_validation=len(val),
                              outer_assessment_loader=False, cuda_initialized=torch.cuda.is_initialized(),
                              max_epochs=50, results=c['results'], checkpoints=c['checkpoints'], signature=signature)))
        return
    require_launch_freeze(c, signature)
    folder, ck = ROOT / c['results'], ROOT / c['checkpoints']
    if args.request_stop:
        if not (ck / 'latest.pt').is_file():
            raise FileNotFoundError('No launched R202 checkpoint exists')
        write_json(folder / 'stop_request.json', dict(experiment_id=c['experiment_id'], reason=args.reason,
                   requested_at=datetime.now(timezone.utc).isoformat(), finish_current_epoch=True))
        print('Stop requested: finish and commit the current epoch; preserve all artifacts')
        return
    with run_lock(c['experiment_id']):
        if args.memory_check:
            memory_check()
        elif args.closeout:
            saved = torch.load(ck / 'latest.pt', map_location='cpu', weights_only=False)
            checked_resume(saved, c, signature)
            _, val, _ = development_data(c)
            closeout(saved, folder, ck, c, val)
            print('Closeout repaired from stored predictions; no CUDA initialization or inference')
        else:
            run(resume=args.resume, allow_user_stop_resume=args.allow_user_stop_resume)


if __name__ == '__main__':
    main()
