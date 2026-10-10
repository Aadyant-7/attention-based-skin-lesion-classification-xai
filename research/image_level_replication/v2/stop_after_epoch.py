"""Administrative stop at a committed R201 epoch, followed by CPU closeout.

This external helper does not amend the loaded training code or its frozen
configuration. It waits for the post-commit EPOCH log, suspends the verified
worker, validates latest.pt on CPU, and terminates only that worker tree.
Any next-epoch work already in flight is uncommitted and discarded.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import time

import psutil

ROOT = Path(__file__).resolve().parents[3]
RUN = 'r201_b3_dermai_adaptation_fold00_seed42'
FOLDER = ROOT / 'results/image_level_replication/v2' / RUN
CK = ROOT / 'checkpoints/image_level_replication/v2' / RUN


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_receipt(value):
    path = FOLDER / 'administrative_stop.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def log(message):
    print(utc(), message, flush=True)


def cpu_closeout(receipt):
    import pandas as pd
    import torch
    from research.common import sha256, relative, write_json
    from research.plots import training_figures
    from .core import config_signature, development_data
    from .train_b3 import (checked_resume, require_launch_freeze, registry_row,
                          update_registries, write_prediction_package, atomic_checkpoint)

    config, signature = config_signature()
    require_launch_freeze(config, signature)
    _, val = development_data(config)
    latest_hash = sha256(CK / 'latest.pt')
    saved = torch.load(CK / 'latest.pt', map_location='cpu', weights_only=False)
    checked_resume(saved, config, signature)
    if saved['epoch'] != receipt['committed_epoch']:
        raise ValueError('Committed epoch changed after administrative stop')
    if saved['epoch'] < receipt['requested_stop_after_epoch']:
        raise ValueError('Requested epoch did not finish')
    original_hashes = {name: sha256(CK / name)
                       for name in ('latest.pt', 'best.pt', 'best_macro_f1.pt')}
    for key, name in (('best_accuracy', 'best.pt'), ('best_macro_f1', 'best_macro_f1.pt')):
        selected = torch.load(CK / name, map_location='cpu', weights_only=False)
        if (selected['epoch'] != saved[key]['epoch'] or
                selected['metrics'] != saved[key]['metrics'] or
                selected['predictions'] != saved[key]['predictions']):
            raise ValueError('Selected checkpoint does not match the committed state: ' + name)
    # The frozen stopping policy remains intact; user cancellation is a separate fact.
    for key, destination in (('best_accuracy', 'best_accuracy'),
                             ('best_macro_f1', 'best_macro_f1'),
                             ('last_validation', 'final')):
        write_prediction_package(saved[key], FOLDER / destination, val, config, figures=True)
    training_figures(pd.DataFrame(saved['history']), FOLDER / 'figures',
                     'B3 fold-00 inner validation — stopped by user', selection_metric='accuracy')
    final_path = CK / 'final.pt'
    if not final_path.exists():
        atomic_checkpoint(final_path, dict(saved, checkpoint_kind='final_committed_epoch',
                                          administrative_stop=receipt))
    else:
        existing = torch.load(final_path, map_location='cpu', weights_only=False)
        if existing['epoch'] != saved['epoch'] or existing.get('administrative_stop') != receipt:
            raise FileExistsError('Existing final.pt differs; original retained')
    # Training already committed its bookkeeping before the EPOCH log. Do not
    # call repair(): it can rewrite a selected checkpoint at the final epoch.
    progress = json.loads((FOLDER / 'progress.json').read_text(encoding='utf-8'))
    progress.update(status='stopped_by_user', stopping_reason='User requested cancellation after the ongoing epoch',
                    training_worker_pid=receipt['worker_pid'], closeout_pid=os.getpid())
    write_json(FOLDER / 'progress.json', progress)
    row = registry_row(saved, config, 'stopped_by_user')
    row['notes'] += ' User cancellation after committed epoch ' + str(saved['epoch']) + '; original stopping policy not amended.'
    update_registries(row)
    write_json(FOLDER / 'record.json', row)
    for name, digest in original_hashes.items():
        if sha256(CK / name) != digest:
            raise ValueError('Original checkpoint changed during CPU closeout: ' + name)
    write_json(FOLDER / 'closeout.json', dict(
        status='stopped_by_user', experiment_id=RUN, committed_epochs=saved['epoch'],
        stopping_reason='User requested cancellation after the ongoing epoch',
        original_meaningful_stopping=saved['meaningful_stopping'],
        administrative_stop=receipt,
        best_accuracy_epoch=saved['best_accuracy']['epoch'],
        best_accuracy=saved['best_accuracy']['metrics'],
        best_macro_f1_epoch=saved['best_macro_f1']['epoch'],
        best_macro_f1=saved['best_macro_f1']['metrics'],
        final_epoch=saved['epoch'], final_metrics=saved['last_validation']['metrics'],
        training_precision=config['training_precision'], validation_precision='fp32',
        numerical_events=saved['numerical_events'], outer_assessment_loaded=False,
        original_checkpoints_unchanged=True, latest_sha256=latest_hash,
        checkpoints={name: dict(path=relative(CK / name), sha256=sha256(CK / name))
                     for name in ('best.pt', 'best_macro_f1.pt', 'latest.pt', 'final.pt')},
        evaluation_claim='Fold-00 inner validation only; not an outer-test or full K10 result'))
    log('CPU closeout complete; original checkpoints preserved; no CUDA or outer inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--closeout-only', action='store_true')
    args = parser.parse_args()
    if args.closeout_only:
        cpu_closeout(json.loads((FOLDER / 'administrative_stop.json').read_text(encoding='utf-8')))
        return
    if not args.pid or not args.epoch or not 1 <= args.epoch <= 50:
        parser.error('Explicit worker PID and epoch (1–50) required')
    if (FOLDER / 'administrative_stop.json').exists():
        raise FileExistsError('Existing stop receipt retained; inspect before retry')
    process = psutil.Process(args.pid)
    command = process.cmdline()
    if 'research.image_level_replication.v2.train_b3' not in command:
        raise ValueError('PID does not belong to the expected training module')
    if Path(process.cwd()).resolve() != ROOT.resolve():
        raise ValueError('Worker is outside the intended repository')
    start = process.create_time()
    receipt = dict(status='waiting_for_committed_epoch', requested_at=utc(),
                   worker_pid=args.pid, worker_created=start, command=command,
                   requested_stop_after_epoch=args.epoch, helper_pid=os.getpid(),
                   user_instruction='Let the ongoing epoch run and then stop',
                   outer_assessment_loaded=False)
    write_receipt(receipt)
    log('Waiting for the post-commit EPOCH ' + str(args.epoch) + ' log; verified PID ' + str(args.pid))
    deadline = time.monotonic() + 3600
    try:
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('No committed boundary within one hour; worker not terminated')
            if not process.is_running() or process.create_time() != start:
                raise RuntimeError('Verified worker exited before the requested stop boundary')
            content = (FOLDER / 'train.log').read_text(encoding='utf-8')
            completed = [int(e) for e in re.findall(r' EPOCH (\d+) accuracy=', content)]
            if completed and max(completed) >= args.epoch:
                process.suspend()
                children = process.children(recursive=True)
                try:
                    for child in children:
                        try:
                            child.suspend()
                        except psutil.NoSuchProcess:
                            pass
                    progress = json.loads((FOLDER / 'progress.json').read_text(encoding='utf-8'))
                    if progress['epoch'] < args.epoch or not (CK / 'latest.pt').is_file():
                        raise ValueError('Log boundary lacks committed progress/checkpoint')
                    # Never terminate a PID after identity changes.
                    if process.create_time() != start:
                        raise ValueError('PID identity changed')
                    receipt.update(status='stopped_at_committed_boundary', stopped_at=utc(),
                                   committed_epoch=progress['epoch'],
                                   worker_children=[child.pid for child in children],
                                   last_log_lines=content.splitlines()[-5:],
                                   next_epoch_uncommitted_work_may_be_discarded=True)
                    for child in children:
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    process.terminate()
                    gone, alive = psutil.wait_procs([process, *children], timeout=10)
                    if alive:
                        raise RuntimeError('Some verified processes did not exit: ' + repr([p.pid for p in alive]))
                    write_receipt(receipt)
                    log('Stopped at committed epoch ' + str(progress['epoch']))
                except Exception:
                    for child in children:
                        try:
                            child.resume()
                        except psutil.NoSuchProcess:
                            pass
                    try:
                        process.resume()
                    except psutil.NoSuchProcess:
                        pass
                    raise
                break
            time.sleep(0.1)
        cpu_closeout(receipt)
    except Exception as error:
        log('STOP/CLOSEOUT ERROR: ' + repr(error))
        write_receipt(dict(receipt, helper_error=repr(error), error_at=utc()))
        raise


if __name__ == '__main__':
    main()
