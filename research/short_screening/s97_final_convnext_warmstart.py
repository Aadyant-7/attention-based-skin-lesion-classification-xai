"""One isolated S79 accuracy-best warm-start branch; --check is CPU-only."""
import argparse
import copy
import gc
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json
from research.registry import upsert
from research.strict_train import atomic_checkpoint, run_lock, runtime_versions
from research.plots import comparison_figures, save
from research.short_screening.lesion_bag_screen import report
from research.final_cbam_development.protocol import (
    Images, verified_development, weights_from_training, make_model, make_optimizer,
    make_scheduler, stage, stopping, check_resume, rng_state, restore_rng,
)
from research.final_cbam_development.runtime import (
    require_launch_freeze, deterministic_cuda, capture_rng, load_rng, loader,
    model_cpu, finite, amp_update, NUMERICS,
)
from research.final_cbam_development.train import validation, checkpoint, selected
from research.final_cbam_development.artifacts import COLS, package

CONFIG = ROOT / 'research/short_screening/s97_final_convnext_warmstart_v1.json'
BASE = ROOT / 'results/short_screening/final_convnext_warmstart_v1'
CODE = [relative(Path(__file__)), 'research/common.py', 'research/registry.py',
        'research/strict_train.py', 'research/models.py', 'research/plots.py',
        'research/short_screening/feature_fusion.py',
        'research/short_screening/lesion_bag_screen.py',
        'research/final_cbam_development/protocol.py',
        'research/final_cbam_development/runtime.py',
        'research/final_cbam_development/train.py',
        'research/final_cbam_development/artifacts.py']


def config_signature():
    c = json.loads(CONFIG.read_text())
    parent, original_signature = require_launch_freeze()
    allowed = {'recipe_version', 'experiment_id', 'question', 'reporting_protocol',
               'initialization', 'max_epochs', 'minimum_epochs', 'learning_rate',
               'fusion_rule', 'fusion_search', 'checkpoints', 'results',
               'training_launch_requires_separate_approval', 'status', 'phase',
               'parent_run', 'parent_epoch', 'parent_checkpoint', 'parent_checkpoint_sha256',
               'parent_history', 'parent_history_sha256', 'parent_predictions',
               'parent_prediction_sha256', 'parent_config', 'parent_config_sha256',
               'parent_latest_checkpoint', 'parent_latest_checkpoint_sha256',
               'epoch_accounting', 'optimizer_initialization', 'scheduler_initialization',
               'seed_initialization', 'extension_amendment', 'auto_next_experiment',
               'excluded_changes', 'test_status'}
    for k in set(parent) | set(c):
        if k not in allowed and c.get(k) != parent.get(k):
            raise ValueError('Unapproved change to parent recipe: ' + k)
    assert c['parent_epoch'] == 33 and c['minimum_epochs'] == 50 and c['max_epochs'] == 70
    assert c['learning_rate'] == {'backbone': 1.5e-5, 'head_attention': 5e-5}
    assert c['test_evaluation'] is False and c['auto_next_experiment'] is False
    for path_key, hash_key in [
        ('parent_checkpoint', 'parent_checkpoint_sha256'),
        ('parent_history', 'parent_history_sha256'),
        ('parent_predictions', 'parent_prediction_sha256'),
        ('parent_config', 'parent_config_sha256'),
        ('parent_latest_checkpoint', 'parent_latest_checkpoint_sha256'),
    ]:
        assert sha256(ROOT / c[path_key]) == c[hash_key], path_key
    reference = ROOT / 'results/short_screening/s83_cbam_f1_addition_cpu'
    sig = dict(config_sha256=sha256(CONFIG), code_sha256={p: sha256(ROOT / p) for p in CODE},
               parent_launch_signature=original_signature,
               reference_plan_sha256=sha256(reference / 'PREDECLARED_PLAN.json'),
               reference_predictions_sha256=sha256(reference / 'validation_predictions.csv'),
               reference_metrics_sha256=sha256(reference / 'validation_metrics.json'))
    return c, sig


def parent_state(c, val):
    """Read latest ONLY for exact ancestry; never import its optimizer/RNG/model40."""
    b = torch.load(ROOT / c['parent_checkpoint'], map_location='cpu', weights_only=False)
    ancestry = torch.load(ROOT / c['parent_latest_checkpoint'], map_location='cpu', weights_only=False)
    epoch = c['parent_epoch']
    assert b['best_epoch'] == epoch and b['selection_metric'] == 'accuracy'
    assert b['class_order'] == list(CLASSES)
    assert b['config'] == json.loads((ROOT / c['parent_config']).read_text())
    assert ancestry['epoch'] == 40 and len(ancestry['history']) == 40
    ancestor_best = ancestry['best_accuracy']
    assert ancestor_best['epoch'] == epoch and ancestor_best['metrics'] == b['metrics']
    assert set(b['model']) == set(ancestor_best['model'])
    assert all(torch.equal(v, ancestor_best['model'][k]) for k, v in b['model'].items())
    history = copy.deepcopy(ancestry['history'][:epoch])
    frame = pd.DataFrame(ancestor_best['predictions'])
    saved = pd.read_csv(ROOT / c['parent_predictions'])
    assert frame.image_id.tolist() == val.image_id.tolist() == saved.image_id.tolist()
    assert frame.true_class.tolist() == val.diagnosis.tolist() == saved.true_class.tolist()
    np.testing.assert_allclose(frame[COLS], saved[COLS], atol=1e-12, rtol=0)
    m = report(val.label.to_numpy(), frame[COLS].to_numpy())
    assert m['confusion_matrix'] == b['metrics']['confusion_matrix']
    assert abs(m['macro_f1'] - b['metrics']['macro_f1']) < 1e-12
    assert history[-1]['val_accuracy'] == b['metrics']['accuracy']
    assert max(r['val_macro_f1'] for r in history) == b['metrics']['macro_f1']
    assert history[-1]['backbone_lr_after'] == c['learning_rate']['backbone']
    assert history[-1]['head_lr_after'] == c['learning_rate']['head_attention']
    assert finite(b['model'])
    best = dict(epoch=epoch, metrics=b['metrics'], predictions=ancestor_best['predictions'], model=b['model'])
    del ancestry
    gc.collect()
    return history, best


def policy(history, c):
    result = stopping(history, c)
    child_reductions = [e for e in result['lr_reduction_epochs'] if e > c['parent_epoch']]
    epoch = len(history)
    settled = bool(child_reductions) and epoch-child_reductions[-1] >= c['stopping']['lr_settle_epochs']
    result.update(child_lr_reduction_epochs=child_reductions, child_lr_settled=settled)
    # An ancestral LR reduction does not count as a fresh scheduler opportunity.
    if epoch < c['max_epochs'] and result['stop'] and not settled:
        result.update(stop=False, reason='running_waiting_for_child_lr_settle')
    return result


def check_payload(payload, c, sig, require_cuda=True):
    ordinary = dict(payload, meaningful_stopping=stopping(payload['history'], c))
    check_resume(ordinary, c, sig)
    if payload['meaningful_stopping'] != policy(payload['history'], c):
        raise ValueError('Warm-start stopping counters differ from committed history')
    if payload['epoch'] < c['parent_epoch'] or payload['new_epochs'] != payload['epoch']-c['parent_epoch']:
        raise ValueError('Invalid branch epoch accounting')
    if require_cuda and 'cuda' not in payload['rng']:
        raise ValueError('Missing CUDA RNG recovery state')
    scheduler = payload['scheduler']
    if scheduler['mode_worse'] != float('-inf') or scheduler['mode'] != 'max':
        raise ValueError('Unexpected scheduler comparison sentinel')
    scheduler_values = {k:v for k,v in scheduler.items() if k!='mode_worse'}
    if not finite([payload['best_accuracy']['model'], payload['best_macro_f1']['model'],
                   payload['scaler'], scheduler_values]):
        raise ValueError('Nonfinite selected model or numerical state')


def seed_scheduler(scheduler, best, c):
    scheduler.best = float(best['metrics']['accuracy'])
    scheduler.num_bad_epochs = 0
    scheduler.last_epoch = c['parent_epoch']


def make_payload(c, sig, model, opt, scheduler, scaler, epoch, history, best, f1, last,
                 updates, runtime, events, consecutive):
    p = checkpoint(c, sig, model, opt, scheduler, scaler, epoch, history, best, f1,
                   last, updates, runtime, events, consecutive)
    p.update(meaningful_stopping=policy(history, c), new_epochs=epoch-c['parent_epoch'],
             inherited_epochs=c['parent_epoch'], optimizer_restart_epoch=c['parent_epoch'],
             initialization=c['initialization'])
    return p


def register(c, payload=None, status='prepared_pending_final_gpu_approval'):
    row = dict(experiment_id=c['experiment_id'], era='structured', record_kind='warmstart_training',
               phase=c['phase'], protocol=c['protocol'], evaluation_split='validation',
               split_manifest=c['split_manifest'], split_sha256=c['split_sha256'],
               method=c['question'], model=c['model'], pretrained_weights=c['weights'],
               attention=c['attention'], seed=c['seed'], image_size=c['image_size'],
               preprocessing=c['preprocessing'], augmentation=c['augmentation'], imbalance=c['imbalance'],
               loss=c['loss'], optimizer=c['optimizer'], batch_size=c['batch_size'],
               backbone_lr=c['learning_rate']['backbone'], head_lr=c['learning_rate']['head_attention'],
               config_path=relative(CONFIG), epochs=0 if payload is None else payload['epoch'],
               status=status, notes='Separate warm-start from S79 accuracy33; fresh optimizer/scheduler/scaler/RNG. '
               'Ancestry33 + at most37 new epochs; minimum17 new before plateau stopping. '
               'Repeated exploratory validation after prior test outcomes; no new test inference. '
               'No epoch/weight/member fusion search or automatic next experiment.')
    if payload is not None:
        b = payload['best_accuracy']
        row.update(best_epoch=b['epoch'], **{k: b['metrics'][k] for k in
                   ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']},
                   val_loss=b['metrics']['loss'], runtime_seconds=payload['runtime_seconds'],
                   history_path=c['results']+'/history.csv', metrics_path=c['results']+'/validation_metrics.json',
                   checkpoint=c['checkpoints']+'/best.pt', checkpoint_available_local=True,
                   checkpoint_sha256=sha256(ROOT/c['checkpoints']/'best.pt'), plots_dir=c['results']+'/figures')
    upsert(row)
    return row


def repair(p, folder, ck, c, status='running', figures=False):
    check_payload(p, c, p['signature'])
    write_csv(folder/'history.csv', p['history'])
    write_csv(folder/'ancestral_history.csv', p['history'][:c['parent_epoch']])
    child = [dict(r, child_epoch=r['epoch']-c['parent_epoch']) for r in p['history'][c['parent_epoch']:]]
    if child:
        write_csv(folder/'child_history.csv', child)
    write_csv(folder/'lr_history.csv', [{k: r[k] for k in
              ['epoch', 'backbone_lr', 'head_lr', 'backbone_lr_after', 'head_lr_after', 'scheduler_lr_reduced']}
              for r in p['history']])
    for key, name, target in [('best_accuracy', 'best.pt', folder),
                              ('best_macro_f1', 'best_macro_f1.pt', folder/'macro_f1_selected')]:
        b = p[key]
        atomic_checkpoint(ck/name, dict(config=c, signature=p['signature'], class_order=list(CLASSES),
                          selection_metric='accuracy' if key=='best_accuracy' else 'macro_f1',
                          best_epoch=b['epoch'], metrics=b['metrics'], model=b['model'],
                          inherited_winner=b['epoch']<=c['parent_epoch']))
        package(target, b['metrics'], b['predictions'], figures,
                'S97 warm-start | exploratory validation | '+key)
    last = p['last_validation']
    package(folder/'final_epoch', last['metrics'], last['predictions'], figures,
            'S97 final branch checkpoint | exploratory validation')
    write_json(folder/'record.json', register(c, p, status))
    write_json(folder/'progress.json', dict(status=status, pid=os.getpid(),
               committed_epoch=p['epoch'], inherited_epochs=c['parent_epoch'], new_epochs=p['new_epochs'],
               meaningful_stopping=p['meaningful_stopping'], checkpoint_created=True,
               first_update_confirmed=p['optimizer_updates']>0, test_loaded=False,
               training_precision='amp_fp16', validation_precision='fp32'))


def closeout(p, folder, ck, c):
    if not p['meaningful_stopping']['stop']:
        raise ValueError('Cannot close out an unfinished branch')
    repair(p, folder, ck, c, status='completed', figures=True)
    import matplotlib.pyplot as plt
    for name, cols, label in [('accuracy_curves', ['train_accuracy', 'val_accuracy'], 'Accuracy'),
                              ('loss_curves', ['train_loss', 'val_loss'], 'Loss'),
                              ('macro_f1_curve', ['val_macro_f1'], 'Macro-F1')]:
        h = pd.DataFrame(p['history'])
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in cols:
            ax.plot(h.epoch, h[col], label=col.replace('_', ' '))
        ax.axvline(c['parent_epoch']+.5, color='black', linestyle='--', label='Optimizer/RNG restart after33')
        ax.set(xlabel='Cumulative branch epoch (1–33 inherited)', ylabel=label,
               title='S97 ancestral trajectory + separate warm-start branch')
        ax.grid(alpha=.2); ax.legend(); save(fig, folder/'figures', name)
    parent_metrics = json.loads((ROOT/c['parent_config']).with_name('validation_metrics.json').read_text())
    comparison_figures([dict(display_name='S79 saved accuracy33', accuracy=parent_metrics['accuracy'], macro_f1=parent_metrics['macro_f1']),
                       dict(display_name='S97 accuracy winner', **{k:p['best_accuracy']['metrics'][k] for k in ['accuracy', 'macro_f1']})],
                       folder/'comparison_figures', 'Standalone exploratory validation; no new ensemble yet')
    summary = dict(status='completed', inherited_epochs=c['parent_epoch'], new_epochs=p['new_epochs'],
                   total_branch_epochs=p['epoch'], best_accuracy_epoch=p['best_accuracy']['epoch'],
                   best_macro_f1_epoch=p['best_macro_f1']['epoch'],
                   best_accuracy=p['best_accuracy']['metrics']['accuracy'],
                   best_macro_f1=p['best_macro_f1']['metrics']['macro_f1'],
                   final_metrics=p['last_validation']['metrics'], stopping=p['meaningful_stopping'],
                   new_optimizer_updates=p['optimizer_updates'], runtime_seconds=p['runtime_seconds'],
                   amp_overflow_events=p['overflow_events'], new_training_only_runtime=True,
                   inherited_accuracy_winner=p['best_accuracy']['epoch']<=c['parent_epoch'],
                   inherited_f1_winner=p['best_macro_f1']['epoch']<=c['parent_epoch'],
                   auto_next_experiment=False, test_loaded=False, ensemble_evaluated=False)
    write_json(folder/'training_summary.json', summary)


def run(resume=False):
    c, sig = config_signature()
    if json.loads((BASE/'launch_manifest.json').read_text()) != sig:
        raise ValueError('CPU preflight freeze changed; recheck before approved launch')
    if json.loads((BASE/'cpu_preflight.json').read_text())['status'] != 'passed':
        raise ValueError('CPU preflight did not pass')
    train, val = verified_development(c)
    weights, _ = weights_from_training(train)
    folder, ck = ROOT/c['results'], ROOT/c['checkpoints']
    latest = ck/'latest.pt'
    saved = None
    if resume:
        if latest.exists():
            saved = torch.load(latest, map_location='cpu', weights_only=False)
            check_payload(saved, c, sig)
            if (folder/'training_summary.json').exists():
                print('Completed S97 preserved; no GPU initialization'); return
        else:
            # A setup failure can happen before the initial checkpoint commit.
            # No optimizer update can precede that commit. Preserve the evidence
            # and reconstruct the unchanged parent initialization explicitly.
            if not folder.exists() or not ck.exists():
                raise FileNotFoundError('No S97 run to recover; use --start')
            if json.loads((folder/'config.json').read_text()) != c or json.loads((folder/'launch_manifest.json').read_text()) != sig:
                raise ValueError('Cannot bootstrap a run with mismatched metadata')
            if any((folder/p).exists() for p in ['history.csv', 'training_summary.json']) or any(ck.glob('*.pt')):
                raise ValueError('Missing latest checkpoint despite committed artifacts; manual recovery required')
            history, best = parent_state(c, val)
            write_json(folder/('setup_recovery_'+str(time.time_ns())+'.json'), dict(
                       recovery='Reconstruct parent33; no initial checkpoint or optimizer update committed',
                       old_evidence_preserved=True, parent_checkpoint_sha256=c['parent_checkpoint_sha256']))
    else:
        if folder.exists() or ck.exists():
            raise FileExistsError('S97 exists; use --resume, never overwrite it')
        history, best = parent_state(c, val)
        folder.mkdir(parents=True); ck.mkdir(parents=True)
        write_json(folder/'config.json', c); write_json(folder/'launch_manifest.json', sig)
        write_json(folder/'environment.json', dict(runtime=runtime_versions(), git_commit=
                   subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    logger = logging.getLogger('s97'); logger.setLevel(logging.INFO)
    handlers = [logging.FileHandler(folder/'train.log'), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s')); logger.addHandler(handler)
    epoch = saved['epoch'] if saved else c['parent_epoch']
    history = saved['history'] if saved else history
    best = saved['best_accuracy'] if saved else best
    f1 = saved['best_macro_f1'] if saved else best
    last = saved['last_validation'] if saved else {k:best[k] for k in ['epoch', 'metrics', 'predictions']}
    updates = saved['optimizer_updates'] if saved else 0
    events = saved['overflow_events'] if saved else []
    consecutive = saved['consecutive_overflows'] if saved else 0
    prior_runtime = saved['runtime_seconds'] if saved else 0.
    tick = time.perf_counter()
    startup_confirmed = False
    try:
        deterministic_cuda(c['seed'])
        model = make_model(c, pretrained=False).cuda()
        model.load_state_dict(saved['model'] if saved else best['model'], strict=True)
        opt = make_optimizer(model, c); scheduler = make_scheduler(opt, c)
        scaler = torch.amp.GradScaler('cuda', init_scale=NUMERICS['gradscaler_initial_scale'])
        weights = weights.cuda()
        if saved:
            opt.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler']); load_rng(saved['rng'])
            repair(saved, folder, ck, c); del saved; gc.collect()
        else:
            seed_scheduler(scheduler, best, c)
            initial = make_payload(c, sig, model, opt, scheduler, scaler, epoch, history, best, f1, last, 0, 0., [], 0)
            atomic_checkpoint(latest, initial); repair(initial, folder, ck, c); del initial
        write_json(folder/'process.json', dict(pid=os.getpid(), resume=resume, committed_epoch=epoch))
        logger.info('LOGGING/CHECKPOINTING ACTIVE resume=%s cumulative_epoch=%d new_epochs=%d; no test loader',
                    resume, epoch, epoch-c['parent_epoch'])
        for next_epoch in range(epoch+1, c['max_epochs']+1):
            if policy(history, c)['stop']: break
            epoch = next_epoch; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            phase = stage(model, epoch, c)
            assert phase == 'full_finetune'
            dl = loader(train, c, epoch, True)
            assert len(dl)%c['gradient_accumulation'] == 0
            iterator = iter(dl); num = den = 0.; correct = used = skipped = 0
            lr_before = [g['lr'] for g in opt.param_groups]
            for i in range(len(dl)//c['gradient_accumulation']):
                batches = []; ids = []
                for _ in range(c['gradient_accumulation']):
                    x, y, batch_ids = next(iterator)
                    batches.append((x.cuda(non_blocking=True), y.cuda(non_blocking=True))); ids.extend(batch_ids)
                stats = amp_update(model, opt, scaler, batches, weights, c)
                if stats['skipped']:
                    skipped += 1; consecutive += 1
                    events.append(dict(epoch=epoch, effective_batch=i+1, image_ids=ids,
                                  scale_before=stats['scale_before'], scale_after=stats['scale_after'],
                                  nonfinite_gradient_parameters=stats['nonfinite_gradient_parameters']))
                    write_json(folder/'amp_overflow_events.json', events)
                    logger.warning('AMP skipped update epoch=%d batch=%d scale=%s->%s',
                                   epoch, i+1, stats['scale_before'], stats['scale_after'])
                    if consecutive >= 3: raise FloatingPointError('Three consecutive gradient overflows; preserve and diagnose')
                else:
                    updates += 1; consecutive = 0
                num += stats['numerator']; den += stats['denominator']; correct += stats['correct']; used += stats['used']
                if not startup_confirmed and not stats['skipped']:
                    startup_confirmed = True
                    write_json(folder/'startup_confirmation.json', dict(pid=os.getpid(), checkpoint_created=latest.exists(),
                               first_update_confirmed=True, cumulative_epoch=epoch, new_epoch=epoch-c['parent_epoch']))
                    logger.info('FIRST ACCEPTED GPU UPDATE; logging and committed checkpoint active')
                if i == 0 or (i+1)%25 == 0:
                    logger.info('epoch %d/%d child=%d update %d/%d', epoch, c['max_epochs'],
                                epoch-c['parent_epoch'], i+1, len(dl)//c['gradient_accumulation'])
            del dl, iterator, batches; gc.collect()
            metrics, p, ids = validation(model, loader(val, c, epoch, False), weights)
            assert ids == val.image_id.tolist()
            frame = val[['image_id', 'lesion_id']].copy(); frame['true_class'] = val.diagnosis
            frame['predicted_class'] = [CLASSES[k] for k in p.argmax(1)]; frame[COLS] = p
            predictions = frame.to_dict('records'); last = dict(epoch=epoch, metrics=metrics, predictions=predictions)
            scheduler.step(metrics['accuracy']); lr_after = [g['lr'] for g in opt.param_groups]
            row = dict(epoch=epoch, epoch_seed=c['seed']+epoch, stage=phase, train_loss=num/den,
                       train_accuracy=correct/used, train_images_used=used, val_loss=metrics['loss'],
                       **{'val_'+k:metrics[k] for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']},
                       backbone_lr=lr_before[0], head_lr=lr_before[1], backbone_lr_after=lr_after[0], head_lr_after=lr_after[1],
                       scheduler_lr_reduced=any(a<b for a,b in zip(lr_after, lr_before)), optimizer_updates=updates,
                       scaler_scale=scaler.get_scale(), amp_skipped_optimizer_updates=skipped,
                       training_precision='amp_fp16', validation_precision='fp32', epoch_seconds=time.perf_counter()-started,
                       peak_allocated_vram_mb=torch.cuda.max_memory_allocated()/2**20)
            history.append(row); state = policy(history, c)
            row.update(meaningful_stale=state['stale'], last_meaningful_epoch=state['last_meaningful_epoch'],
                       meaningful_accuracy_reference=state['references']['accuracy'],
                       meaningful_macro_f1_reference=state['references']['macro_f1'], stopping_reason=state['reason'])
            if metrics['accuracy'] > best['metrics']['accuracy']: best = selected(epoch, metrics, predictions, model)
            if metrics['macro_f1'] > f1['metrics']['macro_f1']: f1 = selected(epoch, metrics, predictions, model)
            payload = make_payload(c, sig, model, opt, scheduler, scaler, epoch, history, best, f1, last,
                                   updates, prior_runtime+time.perf_counter()-tick, events, consecutive)
            atomic_checkpoint(latest, payload)  # Recovery boundary precedes all CSV/registry mutations.
            repair(payload, folder, ck, c); del payload
            logger.info('EPOCH %d child=%d accuracy=%.6f macroF1=%.6f stale=%d stop=%s',
                        epoch, epoch-c['parent_epoch'], metrics['accuracy'], metrics['macro_f1'], state['stale'], state['reason'])
        committed = torch.load(latest, map_location='cpu', weights_only=False)
        check_payload(committed, c, sig); closeout(committed, folder, ck, c)
        logger.info('COMPLETED S97; no automatic fusion, test or next model')
    except BaseException as exc:
        evidence = dict(error=repr(exc), attempted_epoch=epoch, latest_checkpoint=relative(latest),
                        latest_committed_epoch=torch.load(latest, map_location='cpu', weights_only=False)['epoch'] if latest.exists() else None,
                        automatic_retry=False, test_loaded=False)
        stamp = 'failure_'+str(time.time_ns()); write_json(folder/(stamp+'.json'), evidence)
        if isinstance(exc, (FloatingPointError, RuntimeError)) and 'model' in locals() and 'opt' in locals() and 'scaler' in locals():
            atomic_checkpoint(ck/(stamp+'_diagnostic.pt'), dict(diagnostic_only=True, config=c, signature=sig,
                              attempted_epoch=epoch, model=model_cpu(model), optimizer=opt.state_dict(), scaler=scaler.state_dict()))
        write_json(folder/'progress.json', dict(status='failed_preserved', **evidence))
        logger.exception('STOPPED; latest committed branch state preserved'); raise
    finally:
        for handler in handlers: logger.removeHandler(handler); handler.close()


def stopping_checks(history, c):
    def flat_to(epoch, reduction=None):
        h = copy.deepcopy(history)
        for e in range(c['parent_epoch']+1, epoch+1):
            h.append(dict(epoch=e, val_accuracy=history[-1]['val_accuracy'],
                          val_macro_f1=history[-1]['val_macro_f1'], scheduler_lr_reduced=e==reduction))
        return h
    assert not policy(flat_to(49, 37), c)['stop']
    assert not policy(flat_to(50), c)['stop']  # Ancestral LR25 cannot qualify.
    assert policy(flat_to(50, 37), c)['stop']
    assert not policy(flat_to(50, 49), c)['stop']
    assert policy(flat_to(70), c)['reason'] == 'epoch_cap'
    tiny = flat_to(50, 37); tiny[-1]['val_accuracy'] += .0001; tiny[-1]['val_macro_f1'] += .0001
    assert policy(tiny, c)['last_meaningful_epoch'] == 33 and policy(tiny, c)['stop']
    for metric, delta in [('accuracy', .002), ('macro_f1', .003)]:
        h = flat_to(50, 37); h[-1]['val_'+metric] += delta
        assert policy(h, c)['last_meaningful_epoch'] == 50 and not policy(h, c)['stop']
    return 'minimum, meaningful deltas, child LR opportunity/settling and hard cap passed'


def check():
    c, sig = config_signature(); train, val = verified_development(c)
    if (ROOT/c['results']).exists() or (ROOT/c['checkpoints']).exists():
        raise FileExistsError('Do not rewrite the launch freeze after S97 starts')
    torch.set_num_threads(4); torch.manual_seed(c['seed'])
    history, best = parent_state(c, val); guards = stopping_checks(history, c)
    net = make_model(c, pretrained=False); net.load_state_dict(best['model'], strict=True)
    assert stage(net, 34, c) == 'full_finetune' and all(p.requires_grad for p in net.parameters())
    opt = make_optimizer(net, c); scheduler = make_scheduler(opt, c); seed_scheduler(scheduler, best, c)
    assert not opt.state and all(g['weight_decay']==.0001 for g in opt.param_groups)
    weights, counts = weights_from_training(train)
    ds = Images(train.iloc[:2], c, True); samples = [ds[i] for i in range(2)]
    x = torch.stack([a[0] for a in samples]); y = torch.tensor([a[1] for a in samples])
    z = net(x); loss = torch.nn.functional.cross_entropy(z, y, weight=weights)
    loss.backward(); assert finite([z, loss, [p.grad for p in net.parameters()]])
    torch.nn.utils.clip_grad_norm_(net.parameters(), c['gradient_clip_norm']); opt.step()
    assert finite(opt.state_dict())
    try:
        Images(train.iloc[:2].assign(split='test'), c, False)
    except ValueError:
        rejected = True
    else:
        rejected = False
    assert rejected
    payload = dict(config=c, signature=sig, epoch=33, inherited_epochs=33, new_epochs=0,
                   model=model_cpu(net), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), scaler={},
                   rng=rng_state(), history=history, meaningful_stopping=policy(history, c), optimizer_updates=1,
                   best_accuracy=best, best_macro_f1=best, last_validation={k:best[k] for k in ['epoch', 'metrics', 'predictions']})
    (ROOT/'.cache').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='s97_cpu_', dir=ROOT/'.cache') as temp:
        path = Path(temp)/'roundtrip.pt'; atomic_checkpoint(path, payload)
        restored = torch.load(path, map_location='cpu', weights_only=False)
        check_payload(restored, c, sig, require_cuda=False)
        assert all(torch.equal(v, restored['model'][k]) for k,v in payload['model'].items())
        assert restored['optimizer']['param_groups']==payload['optimizer']['param_groups']
        opt.load_state_dict(restored['optimizer']); scheduler.load_state_dict(restored['scheduler'])
        restore_rng(restored['rng']); assert torch.equal(torch.get_rng_state(), restored['rng']['torch'])
    assert not torch.cuda.is_initialized()
    plan = dict(experiment_id=c['experiment_id'], config=c, signature=sig,
                proposal_status='prepared; longer GPU training awaits approval', parent_accuracy=best['metrics']['accuracy'],
                parent_macro_f1=best['metrics']['macro_f1'], inherited_epochs=33, new_epochs_minimum=17, new_epochs_maximum=37,
                frozen_ensemble_reference='S83 equal-six 94.011976% / macro-F1 .895200837',
                reference_plan=json.loads((ROOT/'results/short_screening/s83_cbam_f1_addition_cpu/PREDECLARED_PLAN.json').read_text()),
                post_training_fusion=c['fusion_rule'], maximum_fusions=1, test_loaded=False,
                independent_test_claim=False, additional_net_correct_needed_for_95=15)
    write_json(BASE/'PREDECLARED_PLAN.json', plan); write_json(BASE/'launch_manifest.json', sig)
    write_json(BASE/'cpu_preflight.json', dict(status='passed', train=len(train), validation=len(val),
               class_counts_training=counts, cuda_initialized=False, gpu_training=False, test_loader_rejected=rejected,
               actual_parent_weights_strict_loaded=True, parent_history_exact=True, inherited_f1_uses_only_through33=True,
               fresh_optimizer_finite_update=True, full_state_checkpoint_roundtrip=True, stopping_checks=guards,
               inherited_epochs=33, maximum_new_epochs=37, first_eligible_plateau_epoch=50))
    register(c)
    print(json.dumps(dict(status='prepared_pending_final_gpu_approval', cpu_preflight='passed',
               cuda_initialized=False, inherited_epochs=33, new_epochs_maximum=37, results=c['results'], checkpoints=c['checkpoints'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true'); mode.add_argument('--start', action='store_true')
    mode.add_argument('--resume', action='store_true'); args = parser.parse_args()
    with run_lock(json.loads(CONFIG.read_text())['experiment_id']):
        if args.check: check()
        else: run(args.resume)


if __name__ == '__main__':
    main()
