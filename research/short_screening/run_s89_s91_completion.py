"""Exactly two sequential stages: finish S89 to20, then train S91 Base to20.

This is an independent background helper, not an assistant monitoring loop.
It performs no model selection, ensembling, test inference or third launch.
"""
import json
import logging
import os
import subprocess
import time
from research.common import ROOT, relative, sha256, write_json

OUT=ROOT/'results/short_screening/s89_s91_completion_v1'
STAGES=[
    dict(name='S89 Swin completion',module='research.short_screening.s89_swin_transfer',
         config='research/short_screening/s89_swin_transfer_v1.json',
         result='results/short_screening/swin_transfer_v1/s89_swin_t_none_exploratory_seed42',
         checkpoint='checkpoints/short_screening/swin_transfer_v1/s89_swin_t_none_exploratory_seed42',
         preflight='results/short_screening/swin_transfer_v1/preflight.json'),
    dict(name='S91 DeiT III Base full20',module='research.short_screening.s91_deit3_base_transfer',
         config='research/short_screening/s91_deit3_base_transfer_v1.json',
         result='results/short_screening/deit3_base_transfer_v1/s91_deit3_base_none_exploratory_seed42',
         checkpoint='checkpoints/short_screening/deit3_base_transfer_v1/s91_deit3_base_none_exploratory_seed42',
         preflight='results/short_screening/deit3_base_transfer_v1/preflight.json')]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'master.lock'
    try: handle=lock.open('x')
    except FileExistsError:raise RuntimeError('Master lock exists; verify old process before recovery, never duplicate it')
    try:
        handle.write(str(os.getpid()));handle.flush()
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
            handlers=[logging.FileHandler(OUT/'master.log'),logging.StreamHandler()])
        plans=[]
        for stage in STAGES:
            c=json.loads((ROOT/stage['config']).read_text());p=json.loads((ROOT/stage['preflight']).read_text())
            assert c['max_epochs']==20 and not c['test_evaluation'] and not c['auto_next_experiment']
            assert p['config_sha256']==sha256(ROOT/stage['config'])
            assert all(sha256(ROOT/file)==digest for file,digest in p['fingerprints'].items())
            assert sha256(ROOT/p['pretrained_file'])==p['pretrained_sha256']
            plans.append(dict(**stage,config_sha256=sha256(ROOT/stage['config']),preflight_sha256=sha256(ROOT/stage['preflight'])))
        write_json(OUT/'plan.json',dict(stages=plans,maximum_epochs_each=20,sequential=True,test_loaded=False,
            no_s90_small_training=True,no_ensembles=True,no_automatic_extension=True))
        state=dict(master_pid=os.getpid(),status='running',stages=[],test_loaded=False)
        write_json(OUT/'state.json',state)
        python=ROOT/'.venv/Scripts/python.exe'
        for index,stage in enumerate(STAGES):
            result=ROOT/stage['result'];checkpoint=ROOT/stage['checkpoint']/'latest.pt'
            summary=result/'summary.json'
            if summary.exists():
                saved=json.loads(summary.read_text())
                if saved['status']=='completed':
                    assert saved['epochs']==20
                    logging.info('SKIP completed %s epochs20',stage['name'])
                    state['stages'].append(dict(name=stage['name'],status='skipped_completed',epochs=20));continue
            if index==0:
                assert checkpoint.is_file(), 'S89 must resume saved work, never restart from scratch'
                assert (result/'pilot_epoch15/closeout_verification.json').is_file()
            command=[str(python),'-B','-u','-m',stage['module'],'--run','--stop-after-epoch','20']
            if checkpoint.exists():command.append('--resume')
            checkpoint_before=sha256(checkpoint) if checkpoint.exists() else None
            start=time.time()
            with (OUT/f'stage_{index+1}_console.log').open('ab') as console:
                worker=subprocess.Popen(command,cwd=ROOT,stdout=console,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                item=dict(name=stage['name'],status='running',worker_pid=worker.pid,command=command,
                    resumed=checkpoint_before is not None,checkpoint_before_sha256=checkpoint_before)
                state['stages'].append(item);write_json(OUT/'state.json',state)
                logging.info('START %s PID%s resume=%s stop20 result=%s',stage['name'],worker.pid,item['resumed'],relative(result))
                code=worker.wait()
            if code!=0:
                item.update(status='failed',exit_code=code);state['status']='failed';write_json(OUT/'state.json',state)
                raise RuntimeError(f"{stage['name']} failed; evidence preserved; remaining stage not launched")
            saved=json.loads(summary.read_text());assert saved['status']=='completed' and saved['epochs']==20
            assert (ROOT/stage['checkpoint']/'best.pt').is_file() and (ROOT/stage['checkpoint']/'best_macro_f1.pt').is_file()
            item.update(status='completed',epochs=20,elapsed_seconds=time.time()-start,
                best_accuracy=saved['best_accuracy'],best_macro_f1=saved['best_macro_f1'])
            write_json(OUT/'state.json',state);logging.info('COMPLETE %s epochs20',stage['name'])
        state['status']='completed';write_json(OUT/'state.json',state)
        logging.info('COMPLETE authorized two-stage batch; no third model/ensemble/test run')
    except BaseException as exc:
        write_json(OUT/'failure.json',dict(error=repr(exc),master_pid=os.getpid(),test_loaded=False))
        logging.exception('STOP safely; all stage evidence preserved');raise
    finally:handle.close();lock.unlink(missing_ok=True)


if __name__=='__main__':main()
