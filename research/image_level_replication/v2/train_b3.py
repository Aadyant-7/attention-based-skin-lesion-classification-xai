"""Isolated fold-00 B3 training; never loads an outer assessment image.

--check is CPU-only. Launch/resume requires the frozen preparation signature.
The atomic latest checkpoint, not CSV files, defines a committed epoch.
Partial epochs replay from that boundary; no automatic experiment queue exists.
"""
import argparse
import copy
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json
from research.plots import metric_figures, training_figures, validate_metrics
from research.registry import FIELDS, locked, upsert
from .core import (config_signature, development_data, epoch_loader, make_model,
                   make_optimizer, make_scheduler, stopping, finite,
                   capture_rng, restore_rng, report)

FREEZE = ROOT / 'results/image_level_replication/v2/preparation/b3_fold00_launch_freeze.json'
LOCAL_REGISTRY = ROOT / 'results/image_level_replication/v2/experiment_registry.csv'
PROBABILITY_COLUMNS = ['p_' + name for name in CLASSES]
HISTORY_FIELDS = ['epoch', 'epoch_seed', 'train_loss', 'train_accuracy',
                  'train_images_seen', 'train_images_accepted', 'val_loss',
                  'val_accuracy', 'val_macro_precision', 'val_macro_recall',
                  'val_macro_f1', 'lr', 'lr_after',
                  'scheduler_lr_reduced', 'optimizer_updates',
                  'nonfinite_updates_skipped', 'meaningful_stale',
                  'last_meaningful_epoch', 'meaningful_accuracy_reference',
                  'meaningful_macro_f1_reference', 'stopping_reason',
                  'training_precision', 'validation_precision',
                  'epoch_seconds', 'peak_allocated_vram_mb']


def atomic_checkpoint(path, payload):
    """Write in the destination directory, fsync, then atomically replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


@contextmanager
def run_lock(experiment_id):
    """An OS lock is released even if the process or power supply fails."""
    path = ROOT / '.cache/research_run_locks' / (experiment_id + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def runtime_versions():
    return dict(python=sys.version, **{p: version(p) for p in
                ('torch', 'torchvision', 'numpy', 'pandas', 'Pillow', 'scikit-learn')})


def require_launch_freeze(c, signature):
    if not FREEZE.is_file():
        raise FileNotFoundError('Complete CPU preparation and freeze before launch')
    frozen = json.loads(FREEZE.read_text(encoding='utf-8'))
    if frozen.get('signature') != signature:
        raise ValueError('Config/code/protocol/weights changed after CPU launch freeze')
    for key, prefix in (('results', 'results/image_level_replication/v2/'),
                        ('checkpoints', 'checkpoints/image_level_replication/v2/')):
        path = (ROOT / c[key]).resolve()
        path.relative_to(ROOT)
        if not relative(path).startswith(prefix) or path.name != c['experiment_id']:
            raise ValueError('Run output must be in its own V2 experiment namespace')


def model_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_cpu(value):
    """Snapshot optimizer tensors on CPU without duplicating their CUDA storage."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: state_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [state_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(state_cpu(item) for item in value)
    return copy.deepcopy(value)


def check_scheduler(state, epoch):
    # ReduceLROnPlateau intentionally uses -inf for its max-mode sentinel.
    ordinary = {k: v for k, v in state.items() if k not in ('mode_worse', 'best')}
    if not finite(ordinary) or state.get('mode_worse') != float('-inf'):
        raise FloatingPointError('Invalid scheduler state')
    if epoch and not finite(state.get('best')):
        raise FloatingPointError('Nonfinite scheduler best after validation')
    if not epoch and state.get('best') != float('-inf'):
        raise ValueError('Initial scheduler state is not fresh')


def checked_resume(saved, c, signature):
    """Reject drift and malformed state before initializing CUDA."""
    if saved.get('config') != c or saved.get('signature') != signature:
        raise ValueError('Checkpoint config/signature differs; do not silently amend a run')
    if saved.get('class_order') != list(CLASSES) or saved.get('inference_precision') != 'fp32' or saved.get('outer_assessment_loaded') is not False:
        raise ValueError('Checkpoint class order/precision/partition policy changed')
    if saved.get('runtime_versions') != runtime_versions():
        raise ValueError('Runtime versions differ from the committed run')
    epoch = saved.get('epoch')
    history = saved.get('history')
    if not isinstance(epoch, int) or not isinstance(history, list):
        raise ValueError('Invalid checkpoint epoch/history')
    if not 0 <= epoch <= c['max_epochs'] or [r['epoch'] for r in history] != list(range(1, epoch + 1)):
        raise ValueError('Checkpoint history is not a complete committed epoch sequence')
    if 'rng' not in saved or not saved['rng'].get('cuda'):
        raise ValueError('Checkpoint lacks CUDA RNG state')
    for key in ('model', 'optimizer', 'best_accuracy', 'best_macro_f1', 'last_validation'):
        if not finite(saved.get(key)):
            raise FloatingPointError('Nonfinite committed checkpoint field: ' + key)
    if not finite((history, saved.get('runtime_seconds'), saved.get('numerical_events', []))):
        raise FloatingPointError('Nonfinite committed history/runtime/events')
    check_scheduler(saved['scheduler'], epoch)
    if epoch and (saved.get('last_validation', {}).get('epoch') != epoch):
        raise ValueError('Last validation is not the committed epoch')
    for key in ('best_accuracy', 'best_macro_f1'):
        selected = saved.get(key)
        if epoch and (not selected or not 1 <= selected['epoch'] <= epoch):
            raise ValueError('Missing or invalid raw-best checkpoint selector')
        if epoch:
            metric = 'accuracy' if key == 'best_accuracy' else 'macro_f1'
            expected = max(history, key=lambda r: r['val_' + metric])
            if selected['epoch'] != expected['epoch'] or selected['metrics'][metric] != expected['val_' + metric]:
                raise ValueError('Raw-best selector disagrees with earliest history maximum')
    if saved.get('meaningful_stopping') != stopping(history, c):
        raise ValueError('Meaningful stopping state disagrees with committed history')


def payload(c, signature, model, optimizer, scheduler, epoch, history,
            best, best_f1, last, updates, runtime_seconds, events, consecutive):
    state = dict(config=c, signature=signature, epoch=epoch, model=model_cpu(model),
                 optimizer=state_cpu(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                 rng=capture_rng(cuda=True), history=copy.deepcopy(history),
                 best_accuracy=best, best_macro_f1=best_f1, last_validation=last,
                 meaningful_stopping=stopping(history, c), optimizer_updates=updates,
                 runtime_seconds=runtime_seconds, numerical_events=events,
                 consecutive_nonfinite_updates=consecutive, runtime_versions=runtime_versions(),
                 checkpoint_semantics='Complete epoch boundary; partial epoch replays on resume',
                 class_order=list(CLASSES), inference_precision='fp32',
                 outer_assessment_loaded=False)
    for key in ('model', 'optimizer', 'best_accuracy', 'best_macro_f1', 'last_validation'):
        if not finite(state[key]):
            raise FloatingPointError('Refusing to commit nonfinite checkpoint field: ' + key)
    if not finite((state['history'], state['runtime_seconds'], state['numerical_events'])):
        raise FloatingPointError('Refusing to commit nonfinite history/runtime/events')
    check_scheduler(state['scheduler'], epoch)
    return state


def validate_prediction_record(record, val):
    frame = pd.DataFrame(record['predictions'])
    if frame.image_id.tolist() != val.image_id.tolist() or len(frame) != len(val):
        raise ValueError('Saved validation image order/support changed')
    if frame.true_class.tolist() != val.diagnosis.tolist() or frame.lesion_id.tolist() != val.lesion_id.tolist():
        raise ValueError('Saved validation labels/lesions differ from inner validation')
    probs = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if probs.shape != (len(val), 7) or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError('Invalid saved prediction probabilities')
    if not np.allclose(probs.sum(1), 1, atol=1e-6):
        raise ValueError('Saved probabilities do not sum to one')
    predictions = [CLASSES[i] for i in probs.argmax(1)]
    if frame.predicted_class.tolist() != predictions:
        raise ValueError('Saved class predictions disagree with probabilities')
    recomputed = report(val.label.to_numpy(), probs, record['metrics']['loss'])
    validate_metrics(record['metrics'])
    for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1'):
        if not np.isclose(recomputed[key], record['metrics'][key], atol=1e-10):
            raise ValueError('Saved metrics disagree with probabilities: ' + key)
    if recomputed['confusion_matrix'] != record['metrics']['confusion_matrix']:
        raise ValueError('Saved confusion matrix disagrees with probabilities')
    return frame


def write_prediction_package(record, folder, val, c, figures=False):
    frame = validate_prediction_record(record, val)
    folder.mkdir(parents=True, exist_ok=True)
    metrics = dict(record['metrics'], epoch=record['epoch'], experiment_id=c['experiment_id'],
                   protocol=c['protocol'], evaluation_split='inner_validation',
                   fold=c['fold'], class_order=list(CLASSES), inference_precision='fp32',
                   outer_assessment_loaded=False, correct=int(np.trace(record['metrics']['confusion_matrix'])),
                   incorrect=int(len(frame) - np.trace(record['metrics']['confusion_matrix'])))
    write_json(folder / 'validation_metrics.json', metrics)
    write_csv(folder / 'predictions.csv', frame.to_dict('records'))
    write_csv(folder / 'probabilities.csv', frame[['image_id', 'lesion_id', 'true_class', *PROBABILITY_COLUMNS]].to_dict('records'))
    write_csv(folder / 'per_class_metrics.csv', [{'class': k, **metrics['per_class'][k]} for k in CLASSES])
    cm = np.asarray(metrics['confusion_matrix'])
    norm = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(1, keepdims=True) > 0)
    for name, matrix in (('confusion_matrix', cm), ('confusion_matrix_normalized', norm)):
        write_csv(folder / (name + '.csv'), [{'true_class': k, **dict(zip(CLASSES, matrix[i].tolist()))} for i, k in enumerate(CLASSES)])
    if figures:
        metric_figures(metrics, folder / 'figures', 'B3 fold-00 inner validation — epoch ' + str(record['epoch']))


def selected_checkpoint(selected, c, signature, kind):
    return dict(config=c, signature=signature, epoch=selected['epoch'], model=selected['model'],
                metrics=selected['metrics'], predictions=selected['predictions'], checkpoint_kind=kind,
                selection_partition='inner_validation', inference_precision='fp32', class_order=list(CLASSES),
                outer_assessment_loaded=False)


def registry_row(saved, c, status):
    row = {key: '' for key in FIELDS}
    best = saved.get('best_accuracy')
    row.update(experiment_id=c['experiment_id'], era='structured', record_kind='training_run',
               phase='image_level_replication_v2', protocol=c['protocol'], evaluation_split='inner_validation_fold00',
               split_manifest=c['split_manifest'], split_sha256=c['split_sha256'], method='DermAI-inspired plain B3 adaptation',
               model=c['model'], pretrained_weights=c['weights'], attention=c['attention'], image_size=c['image_size'],
               preprocessing=json.dumps(c['preprocessing'], sort_keys=True), augmentation=json.dumps(c['augmentation'], sort_keys=True),
               imbalance=json.dumps(c['imbalance'], sort_keys=True), loss=c['loss'], optimizer=c['optimizer'],
               backbone_lr=c['learning_rate'], head_lr=c['learning_rate'], batch_size=c['batch_size'], seed=c['seed'],
               epochs=saved['epoch'], runtime_seconds=saved['runtime_seconds'], config_path=c['config_path'],
               history_path=c['results'] + '/history.csv', status=status,
               decision='inner_validation_development_only',
               notes='Fresh ImageNet start; post-development image-level fold 00; natural lesion overlap; no outer assessment inference. '
                     'Meaningful stopping distinguishes raw numerical best checkpoints from patience resets.')
    if best:
        row.update(best_epoch=best['epoch'], **{k: best['metrics'][k] for k in
                   ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')}, val_loss=best['metrics']['loss'],
                   checkpoint=c['checkpoints'] + '/best.pt', checkpoint_available_local=True,
                   metrics_path=c['results'] + '/best_accuracy/validation_metrics.json',
                   confusion_matrix_path=c['results'] + '/best_accuracy/confusion_matrix.csv',
                   plots_dir=c['results'] + '/best_accuracy/figures')
        path = ROOT / row['checkpoint']
        if path.is_file():
            row['checkpoint_sha256'] = sha256(path)
    return row


def update_registries(row):
    """Only this new ID is upserted; no historical rows are rebuilt."""
    with locked(LOCAL_REGISTRY):
        if LOCAL_REGISTRY.is_file():
            import csv
            with LOCAL_REGISTRY.open(encoding='utf-8', newline='') as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != FIELDS:
                    raise ValueError('V2 registry schema changed')
                rows = list(reader)
        else:
            rows = []
        if any(None in r for r in rows) or len({r['experiment_id'] for r in rows}) != len(rows):
            raise ValueError('Malformed V2 registry or duplicate experiment ID')
        previous = next((r for r in rows if r['experiment_id'] == row['experiment_id']), None)
        if previous and (previous['era'] != row['era'] or previous['protocol'] != row['protocol']):
            raise ValueError('Cannot overwrite another protocol/era')
        rows = [r for r in rows if r['experiment_id'] != row['experiment_id']] + [row]
        write_csv(LOCAL_REGISTRY, sorted(rows, key=lambda r: r['experiment_id']), FIELDS)
    upsert(row)


def repair(saved, folder, ck, c, val, status='running', force_selected=False):
    """Reconstruct derived bookkeeping using the last committed checkpoint."""
    write_csv(folder / 'history.csv', saved['history'], HISTORY_FIELDS)
    write_csv(folder / 'lr_history.csv', [{'epoch': r['epoch'], 'lr_before': r['lr'],
              'lr_after': r['lr_after'], 'reduced': r['scheduler_lr_reduced']} for r in saved['history']],
              ['epoch', 'lr_before', 'lr_after', 'reduced'])
    write_json(folder / 'meaningful_stopping.json', saved['meaningful_stopping'])
    write_json(folder / 'numerical_events.json', saved['numerical_events'])
    write_json(folder / 'progress.json', dict(status=status, pid=os.getpid(), epoch=saved['epoch'],
               max_epochs=c['max_epochs'], best_accuracy_epoch=(saved['best_accuracy'] or {}).get('epoch'),
               best_macro_f1_epoch=(saved['best_macro_f1'] or {}).get('epoch'),
               meaningful_stopping=saved['meaningful_stopping'], runtime_seconds=saved['runtime_seconds'],
               results=c['results'], checkpoints=c['checkpoints'], outer_assessment_loaded=False))
    for key, name, destination in (('best_accuracy', 'best.pt', 'best_accuracy'),
                                    ('best_macro_f1', 'best_macro_f1.pt', 'best_macro_f1')):
        selected = saved.get(key)
        if selected:
            if force_selected or selected['epoch'] == saved['epoch'] or not (ck / name).is_file():
                atomic_checkpoint(ck / name, selected_checkpoint(selected, c, saved['signature'], key))
            write_prediction_package(selected, folder / destination, val, c)
    if saved.get('last_validation'):
        write_prediction_package(saved['last_validation'], folder / 'final', val, c)
        write_prediction_package(saved['best_accuracy'], folder, val, c)
    row = registry_row(saved, c, status)
    update_registries(row)
    write_json(folder / 'record.json', row)


def closeout(saved, folder, ck, c, val):
    if not saved['epoch'] or saved['last_validation'] is None:
        raise ValueError('No committed validation epoch exists to close out')
    policy = stopping(saved['history'], c)
    if saved['epoch'] < c['max_epochs'] and not policy['stop']:
        raise ValueError('Unfinished run has not met a stopping rule; use --resume')
    repair(saved, folder, ck, c, val, status='closing_out', force_selected=True)
    atomic_checkpoint(ck / 'final.pt', dict(saved, checkpoint_kind='final_committed_epoch'))
    for key, destination in (('best_accuracy', 'best_accuracy'), ('best_macro_f1', 'best_macro_f1'),
                              ('last_validation', 'final')):
        write_prediction_package(saved[key], folder / destination, val, c, figures=True)
    training_figures(pd.DataFrame(saved['history']), folder / 'figures',
                     'B3 fold-00 development — inner validation only', selection_metric='accuracy')
    write_csv(folder / 'lr_history.csv', [{'epoch': r['epoch'], 'lr_before': r['lr'],
              'lr_after': r['lr_after'], 'reduced': r['scheduler_lr_reduced']} for r in saved['history']])
    write_json(folder / 'closeout.json', dict(status='completed', experiment_id=c['experiment_id'],
               committed_epochs=saved['epoch'], stopping=policy,
               best_accuracy_epoch=saved['best_accuracy']['epoch'], best_accuracy=saved['best_accuracy']['metrics'],
               best_macro_f1_epoch=saved['best_macro_f1']['epoch'], best_macro_f1=saved['best_macro_f1']['metrics'],
               final_epoch=saved['epoch'], final_metrics=saved['last_validation']['metrics'],
               training_precision=c['training_precision'], validation_precision='fp32',
               numerical_events=saved['numerical_events'], outer_assessment_loaded=False,
               checkpoints={n: dict(path=relative(ck / n), sha256=sha256(ck / n)) for n in
                            ('best.pt', 'best_macro_f1.pt', 'latest.pt', 'final.pt')},
               evaluation_claim='Fold-00 inner validation only; not outer/held-out test or full K10'))
    repair(saved, folder, ck, c, val, status='completed')


def validation(model, loader):
    """Identity FP32 validation only; no alternate view or precision search."""
    model.eval()
    labels, ids, arrays = [], [], []
    numerator = total = 0
    with torch.inference_mode():
        for images, targets, image_ids in loader:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            if not torch.isfinite(images).all():
                raise FloatingPointError('Nonfinite FP32 validation input; IDs=' + repr(list(image_ids)))
            logits = model(images).float()
            probabilities = logits.softmax(1)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            if not finite((logits, probabilities, loss)):
                raise FloatingPointError('Nonfinite FP32 validation logits/probabilities/loss; IDs=' + repr(list(image_ids)))
            numerator += float(loss)
            total += len(targets)
            labels.extend(targets.cpu().tolist())
            ids.extend(image_ids)
            arrays.append(probabilities.cpu().numpy())
    if total == 0:
        raise ValueError('Empty validation loader')
    probs = np.concatenate(arrays)
    return report(np.asarray(labels), probs, numerator / total), probs, ids


def selected(epoch, metrics, predictions, model):
    return dict(epoch=epoch, metrics=metrics, predictions=predictions, model=model_cpu(model))


def run(resume=False):
    c, signature = config_signature()
    require_launch_freeze(c, signature)
    train, val = development_data(c)
    folder, ck = ROOT / c['results'], ROOT / c['checkpoints']
    latest = ck / 'latest.pt'
    saved = None
    if resume:
        if not latest.is_file():
            raise FileNotFoundError('No atomic latest checkpoint exists; inspect preserved failure evidence')
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        checked_resume(saved, c, signature)
        if saved['epoch'] >= c['max_epochs'] or stopping(saved['history'], c)['stop']:
            closeout(saved, folder, ck, c, val)
            print('Run already at a stopping boundary; closeout repaired without CUDA initialization')
            return
    else:
        if folder.exists() or ck.exists():
            raise FileExistsError('This experiment already exists; use explicit --resume')
        folder.mkdir(parents=True)
        ck.mkdir(parents=True)
        write_json(folder / 'config.json', c)
        write_json(folder / 'launch_manifest.json', signature)
        write_json(folder / 'environment.json', dict(runtime=runtime_versions(),
                   git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    logger = logging.getLogger(c['experiment_id'])
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(folder / 'train.log', encoding='utf-8'), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    clock = time.perf_counter()
    base_runtime = (saved or {}).get('runtime_seconds', 0)
    epoch = (saved or {}).get('epoch', 0)
    history = (saved or {}).get('history', [])
    best = (saved or {}).get('best_accuracy')
    best_f1 = (saved or {}).get('best_macro_f1')
    last = (saved or {}).get('last_validation')
    updates = (saved or {}).get('optimizer_updates', 0)
    events = (saved or {}).get('numerical_events', [])
    consecutive = (saved or {}).get('consecutive_nonfinite_updates', 0)
    try:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG', c['cuda_workspace_config']) != c['cuda_workspace_config']:
            raise ValueError('CUBLAS workspace setting differs from the frozen deterministic config')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = c['cuda_workspace_config']
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device required; --check remains CPU-only')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('Frozen recipe requires BF16 support; no silent precision substitution')
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
        model = make_model(c, pretrained=not resume).cuda()
        optimizer = make_optimizer(model, c)
        scheduler = make_scheduler(optimizer, c)
        if saved:
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            restore_rng(saved['rng'])
            repair(saved, folder, ck, c, val, force_selected=True)
        else:
            saved = payload(c, signature, model, optimizer, scheduler, 0, [], None,
                            None, None, 0, 0, [], 0)
            atomic_checkpoint(latest, saved)
            repair(saved, folder, ck, c, val)
        del saved
        write_json(folder / 'process.json', dict(pid=os.getpid(), resume=resume, committed_epoch=epoch,
                   results=c['results'], checkpoints=c['checkpoints'], outer_assessment_loaded=False))
        logger.info('LOGGING/CHECKPOINTING ACTIVE resume=%s committed_epoch=%d train=%d inner_val=%d latest=%s; no outer/test loader',
                    resume, epoch, len(train), len(val), relative(latest))
        first_accepted = False
        for epoch in range(epoch + 1, c['max_epochs'] + 1):
            if stopping(history, c)['stop']:
                break
            epoch_clock = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            loader = epoch_loader(train, c, epoch, True)
            numerator = correct = accepted = seen = skipped = 0
            lr_before = optimizer.param_groups[0]['lr']
            for step, (images, targets, image_ids) in enumerate(loader, 1):
                diagnostic_batch = dict(epoch=epoch, step=step, image_ids=list(image_ids))
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                seen += len(targets)
                if not torch.isfinite(images).all():
                    raise FloatingPointError('Nonfinite training input; IDs=' + repr(list(image_ids)))
                optimizer.zero_grad(set_to_none=True)
                # Restore mutable BN buffers if a gradient update must be skipped.
                buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(images)
                loss = F.cross_entropy(logits.float(), targets)
                if not finite((logits, loss)):
                    raise FloatingPointError('Nonfinite BF16 training logits/loss; IDs=' + repr(list(image_ids)))
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip_norm'], error_if_nonfinite=False)
                if not torch.isfinite(norm):
                    bad = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                    with torch.no_grad():
                        for name, buffer in model.named_buffers():
                            buffer.copy_(buffers[name])
                    optimizer.zero_grad(set_to_none=True)
                    consecutive += 1
                    skipped += 1
                    event = dict(epoch=epoch, step=step, image_ids=list(image_ids),
                                 kind='nonfinite_gradient_update_skipped', nonfinite_parameters=bad,
                                 consecutive=consecutive, batchnorm_buffers_restored=True)
                    events.append(event)
                    write_json(folder / 'numerical_events_live.json', events)
                    logger.warning('NONFINITE GRADIENT update skipped epoch=%d step=%d consecutive=%d', epoch, step, consecutive)
                    if consecutive >= c['max_consecutive_nonfinite_updates']:
                        raise FloatingPointError('Consecutive nonfinite update limit reached; latest valid epoch preserved')
                    continue
                optimizer.step()
                updates += 1
                consecutive = 0
                numerator += float(loss.detach()) * len(targets)
                correct += int((logits.detach().argmax(1) == targets).sum())
                accepted += len(targets)
                if not first_accepted:
                    first_accepted = True
                    write_json(folder / 'first_accepted_update.json', dict(epoch=epoch, step=step,
                               optimizer_updates=updates, committed_checkpoint=relative(latest),
                               checkpoint_exists=latest.is_file(), log_active=True))
                    logger.info('FIRST ACCEPTED GPU UPDATE; log active and committed recovery checkpoint exists')
                if step == 1 or step % 100 == 0 or step == len(loader):
                    logger.info('epoch %d/%d step %d/%d CE=%.6f lr=%.7g', epoch, c['max_epochs'], step, len(loader), float(loss.detach()), lr_before)
            if not accepted:
                raise FloatingPointError('Epoch had no accepted optimizer updates')
            metrics, probs, ids = validation(model, epoch_loader(val, c, epoch, False))
            if ids != val.image_id.tolist():
                raise ValueError('Validation IDs/order do not match frozen inner partition')
            predictions = val[['image_id', 'lesion_id']].copy()
            predictions['true_class'] = val.diagnosis
            predictions['predicted_class'] = [CLASSES[i] for i in probs.argmax(1)]
            predictions[PROBABILITY_COLUMNS] = probs
            last = dict(epoch=epoch, metrics=metrics, predictions=predictions.to_dict('records'))
            scheduler.step(metrics['accuracy'])
            lr_after = optimizer.param_groups[0]['lr']
            row = dict(epoch=epoch, epoch_seed=c['seed'] + epoch * 1009,
                       train_loss=numerator / accepted, train_accuracy=correct / accepted,
                       train_images_seen=seen, train_images_accepted=accepted,
                       val_loss=metrics['loss'], **{'val_' + k: metrics[k] for k in
                        ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')},
                       lr=lr_before, lr_after=lr_after,
                       scheduler_lr_reduced=lr_after < lr_before, optimizer_updates=updates,
                       nonfinite_updates_skipped=skipped, training_precision=c['training_precision'],
                       validation_precision='fp32', epoch_seconds=time.perf_counter() - epoch_clock,
                       peak_allocated_vram_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)
            history.append(row)
            policy = stopping(history, c)
            row.update(meaningful_stale=policy['stale'], last_meaningful_epoch=policy['last_meaningful_epoch'],
                       meaningful_accuracy_reference=policy['accuracy_reference'],
                       meaningful_macro_f1_reference=policy['macro_f1_reference'], stopping_reason=policy['reason'])
            if best is None or metrics['accuracy'] > best['metrics']['accuracy']:
                best = selected(epoch, metrics, last['predictions'], model)
            if best_f1 is None or metrics['macro_f1'] > best_f1['metrics']['macro_f1']:
                best_f1 = selected(epoch, metrics, last['predictions'], model)
            committed = payload(c, signature, model, optimizer, scheduler, epoch, history, best,
                                best_f1, last, updates, base_runtime + time.perf_counter() - clock, events, consecutive)
            atomic_checkpoint(latest, committed)  # Always before CSV / registry mutations.
            repair(committed, folder, ck, c, val)
            logger.info('EPOCH %d accuracy=%.6f macroF1=%.6f best_accuracy_epoch=%d best_F1_epoch=%d meaningful_stale=%d stop=%s',
                        epoch, metrics['accuracy'], metrics['macro_f1'], best['epoch'], best_f1['epoch'], policy['stale'], policy['reason'])
            del committed, loader
            if policy['stop']:
                break
        committed = torch.load(latest, map_location='cpu', weights_only=False)
        checked_resume(committed, c, signature)
        closeout(committed, folder, ck, c, val)
        logger.info('COMPLETED fold-00 inner development at epoch=%d; no outer/test inference or next experiment', committed['epoch'])
    except BaseException as exc:
        stamp = 'failure_' + str(time.time_ns())
        committed_epoch = None
        committed = None
        secondary_errors = []
        if latest.is_file():
            try:
                committed = torch.load(latest, map_location='cpu', weights_only=False)
                committed_epoch = committed['epoch']
            except Exception as checkpoint_error:
                secondary_errors.append('Reading latest: ' + repr(checkpoint_error))
        info = dict(error=repr(exc), attempted_epoch=epoch, latest_committed_epoch=committed_epoch,
                    latest_valid_checkpoint=relative(latest) if latest.is_file() else None,
                    automatic_retry=False, outer_assessment_loaded=False,
                    batch=locals().get('diagnostic_batch'), secondary_errors=secondary_errors)
        write_json(folder / (stamp + '.json'), info)
        if 'model' in locals():
            try:
                atomic_checkpoint(ck / (stamp + '_diagnostic.pt'), dict(diagnostic_only=True,
                                  config=c, signature=signature, attempted_epoch=epoch,
                                  model=model_cpu(model), optimizer=state_cpu(optimizer.state_dict()) if 'optimizer' in locals() else None,
                                  gradients={name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None},
                                  numerical_events=events, batch=locals().get('diagnostic_batch'),
                                  inference_or_resume_eligible=False))
            except Exception as diagnostic_error:
                secondary_errors.append('Saving diagnostics: ' + repr(diagnostic_error))
        if committed is not None:
            try:
                repair(committed, folder, ck, c, val, status='failed_preserved', force_selected=True)
            except Exception as bookkeeping_error:
                secondary_errors.append('Failure bookkeeping: ' + repr(bookkeeping_error))
        write_json(folder / (stamp + '.json'), info)
        write_json(folder / 'progress.json', dict(status='failed_preserved', **info))
        logger.exception('STOPPED; committed checkpoints and failure evidence preserved')
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Only the frozen fold-00 config is accepted')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--closeout', action='store_true')
    args = parser.parse_args()
    c, signature = config_signature()
    if args.config and (ROOT / args.config).resolve() != (ROOT / c['config_path']).resolve():
        raise ValueError('Alternate configs are not accepted by this isolated experiment')
    if args.check:
        train, val = development_data(c)
        print(json.dumps(dict(status='cpu_preflight_only_pending_gpu_approval', train=len(train),
                              inner_validation=len(val), outer_assessment_loader=False,
                              cuda_initialized=torch.cuda.is_initialized(), max_epochs=c['max_epochs'],
                              results=c['results'], checkpoints=c['checkpoints'], signature=signature)))
        return
    require_launch_freeze(c, signature)
    with run_lock(c['experiment_id']):
        if args.closeout:
            _, val = development_data(c)
            saved = torch.load(ROOT / c['checkpoints'] / 'latest.pt', map_location='cpu', weights_only=False)
            checked_resume(saved, c, signature)
            closeout(saved, ROOT / c['results'], ROOT / c['checkpoints'], c, val)
            print('Closeout repaired from committed state; no CUDA initialization or inference')
        else:
            run(resume=args.resume)


if __name__ == '__main__':
    main()
