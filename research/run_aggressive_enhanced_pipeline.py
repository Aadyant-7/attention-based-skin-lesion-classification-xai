"""One authorized post-test enhanced queue; immutable protocol, safe recovery, fixed fusion/XAI, STOP."""
import json,logging,os,subprocess,sys,time
from pathlib import Path
import psutil,torch
from research.common import ROOT,write_json,sha256
from research.train import run_lock,runtime_versions,tensor_nonfinite_names
from research.registry import read_registry,REGISTRY
from research.aggressive.core import OUT,CKPT,CONFIG,config,hashes,verify_saved
from research.aggressive.core import compatible_sources
from research.aggressive.stopping import stopping_status

def record_stopping_audit(rid):
    import pandas as pd
    folder=OUT/rid
    decision=stopping_status(pd.read_csv(folder/'history.csv').to_dict('records'))
    write_json(folder/'early_stopping_state.json',decision)
    write_json(CKPT/rid/'meaningful_stopping_audit.json',decision)
    summary_path=folder/'training_summary.json'
    summary=json.loads(summary_path.read_text())
    if 'meaningful_stopping' not in summary:
        summary.update(meaningful_stopping=decision,original_stop_reason=summary.get('stop_reason'),
            policy_application='Legacy live worker stopped naturally; meaningful counters replayed without modifying checkpoints or history')
        write_json(summary_path,summary)

def amended_boundary_handoff(s,rid,log):
    """One authorized legacy-worker handoff; never touch an uncommitted model state."""
    amendment_path=OUT/'stopping_policy_amendment.json'
    if not amendment_path.exists():return False
    amendment=json.loads(amendment_path.read_text())
    if s.get('pid')!=amendment.get('legacy_worker_pid') or amendment.get('handoff_complete'):return False
    folder=OUT/rid
    if (folder/'record.json').exists():return False
    history_path=folder/'history.csv'
    if not history_path.exists():return False
    import pandas as pd,shutil
    history=pd.read_csv(history_path).to_dict('records');decision=stopping_status(history)
    write_json(folder/'early_stopping_state.json',decision)
    if not decision['stop']:return False
    checkpoint=CKPT/rid/'latest.pt'
    committed=torch.load(checkpoint,map_location='cpu',weights_only=False)
    expected_config=config();expected_spec=next(m for m in expected_config['models'] if m['id']==rid)
    if len(committed['history'])!=len(history) or not compatible(spec=expected_spec,c=expected_config):return False
    # The checkpoint and CSV agree on a committed epoch. Any subsequent partial
    # epoch is discarded, never a completed epoch. No unsafe live-code injection.
    process=psutil.Process(s['pid'])
    if abs(process.create_time()-s['create_time'])>.01:raise RuntimeError('Worker identity changed')
    children=process.children(recursive=True)
    for child in reversed(children):
        try:child.terminate()
        except psutil.NoSuchProcess:pass
    process.terminate();_,survivors=psutil.wait_procs(children+[process],timeout=10)
    for survivor in survivors:
        try:survivor.kill()
        except psutil.NoSuchProcess:pass
    _,survivors=psutil.wait_procs(survivors,timeout=10)
    if survivors:raise RuntimeError('Worker did not stop; checkpoints not moved')
    archive=CKPT/rid/'stopping_policy_v1_preserved';archive.mkdir(exist_ok=False)
    for name in ['latest.pt','best.pt','best_macro_f1.pt']:
        source=CKPT/rid/name
        if source.exists():shutil.copy2(source,archive/name)
    shutil.copy2(history_path,archive/'history.csv')
    amendment.update(handoff_complete=True,committed_epoch=len(committed['history']),decision=decision,
        handoff='Committed checkpoint continuation for closeout; no completed epoch repeated',
        checkpoint_sha256=sha256(checkpoint),partial_epoch_discarded=True)
    write_json(amendment_path,amendment)
    log.info('Meaningful stop boundary epoch %d; preserved v1 checkpoints; next worker closes out without another training epoch',len(committed['history']))
    return True

STATE=OUT/'status.json'

def planned_handoff_pending(rid):
    p=OUT/'stopping_policy_amendment.json'
    if not p.exists():return False
    a=json.loads(p.read_text())
    return bool(a.get('handoff_complete') and a.get('decision',{}).get('stop')
                and rid=='s32_efficientnet_b3_cbam_enhanced_seed42')

def alive(s):
    try:p=psutil.Process(s['pid']);return p.is_running() and abs(p.create_time()-s['create_time'])<1
    except (KeyError,psutil.NoSuchProcess):return False

def compatible(spec,c):
    k=torch.load(CKPT/spec['id']/'latest.pt',map_location='cpu',weights_only=False)
    ok=k['config']==c and k['spec']==spec and compatible_sources(k['source_hashes']) and k['runtime_versions']==runtime_versions()
    ok=ok and [r['epoch'] for r in k['history']]==list(range(1,len(k['history'])+1)) and len(k['history'])<=50
    ok=ok and not tensor_nonfinite_names(k['model']) and not tensor_nonfinite_names(k['optimizer'])
    return ok

def preserved():
    original=json.loads((OUT/'historical_preservation.json').read_text())
    for name,digest in original['files'].items():
        if sha256(ROOT/name)!=digest:raise RuntimeError('Historical artifact changed: '+name)
    current={r['experiment_id']:r for r in read_registry()}
    if any(current.get(r['experiment_id'])!=r for r in original['baseline_registry']):raise RuntimeError('Historical registry row changed')
    freeze=json.loads((ROOT/'results/final_strict/v1/ensemble/frozen_ensemble.json').read_text())
    for m in freeze['members']:
        if sha256(ROOT/m['checkpoint'])!=m['sha256']:raise RuntimeError('Original frozen checkpoint changed')

def publish(c,log):
    paths=['research/aggressive','research/run_aggressive_enhanced_pipeline.py','results/aggressive_enhanced','results/master_experiment_registry.csv']
    try:
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='structured-research':raise RuntimeError('Branch changed')
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:raise RuntimeError('Unrelated staged changes')
        subprocess.run(['git','add','--']+paths,cwd=ROOT,check=True,timeout=60)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:subprocess.run(['git','commit','-m','Close predeclared enhanced development pipeline and XAI; preserve original held-out result'],cwd=ROOT,check=True,timeout=60)
        subprocess.run(['git','push','origin','structured-research'],cwd=ROOT,check=True,timeout=60)
        write_json(OUT/'publication.json',dict(status='pushed'));log.info('Final enhanced artifacts pushed')
    except Exception as e:write_json(OUT/'publication.json',dict(status='pending',reason=repr(e)));log.warning('Local artifacts complete; publication deferred: %s',e)

def stale_lock(log):
    lock=REGISTRY.with_suffix('.lock')
    if not lock.exists():return
    for p in psutil.process_iter(['cmdline','cwd']):
        try:
            if p.info['cwd'] and Path(p.info['cwd']).resolve()==ROOT and any(s in ' '.join(p.info['cmdline'] or []) for s in ['research.aggressive.train','research.strict_train','-m research.train']):return
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    lock.unlink();log.warning('Orphaned registry lock released; no training writer observed')

def main():
    budget=OUT/'budget_control.json'
    if budget.exists() and json.loads(budget.read_text()).get('automatic_training_disabled'):
        print('Long-run queue disabled by budget decision; use CPU screening report, not this launcher.',flush=True)
        return
    c=config();pre=json.loads((OUT/'preflight_complete.json').read_text())
    if pre['status']!='passed' or pre['config_sha256']!=sha256(CONFIG):raise RuntimeError('Prepared GPU preflight does not match recipe')
    if [s['model'] for s in c['models']]!=['efficientnet_b3','densenet201','resnet101'] or c['models'][0]['attention']!='cbam' or c['primary_weights']!=[.4,.4,.2]:raise ValueError('Fixed architecture/fusion changed')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'master.log',encoding='utf-8'),logging.StreamHandler()]);log=logging.getLogger('enhanced_queue')
    with run_lock('aggressive_enhanced_master_v1'):
        state=json.loads(STATE.read_text()) if STATE.exists() else dict(status='initialized',models={s['id']:dict(status='pending',attempts=0,fp32_recovery=False) for s in c['models']},source_hashes=hashes())
        if not compatible_sources(state['source_hashes']):raise RuntimeError('Recipe/code changed; no silent resume')
        if state['source_hashes']!=hashes():
            log.info('Authorized stopping-policy-only amendment; original checkpoints/history retained')
            state['source_hashes']=hashes()
        preserved()
        state.update(master_pid=os.getpid(),create_time=psutil.Process().create_time(),execution_mode='sequential');write_json(STATE,state)
        if state['status']=='completed':publish(c,log);return
        try:
            for spec in c['models']:
                rid=spec['id'];s=state['models'][rid];folder=OUT/rid;record=folder/'record.json';latest=CKPT/rid/'latest.pt'
                if record.exists() and json.loads(record.read_text())['status']=='completed':
                    verify_saved(rid,c);record_stopping_audit(rid);s['status']='completed';write_json(STATE,state);log.info('VERIFIED/SKIP %s',rid);continue
                if s['status']=='failed':
                    if planned_handoff_pending(rid) and s.get('pid')==json.loads((OUT/'stopping_policy_amendment.json').read_text()).get('legacy_worker_pid') and latest.exists() and compatible(spec,c):
                        s['status']='pending_closeout';write_json(STATE,state)
                        log.info('Recover planned epoch-boundary closeout; recorded failure evidence remains preserved')
                    else:raise RuntimeError('Unrecoverable stage preserved; manual diagnosis required: '+rid)
                while True:
                    proc=None
                    if alive(s):log.info('Reattached live %s PID %s',rid,s['pid'])
                    else:
                        resume=latest.exists()
                        if resume and not compatible(spec,c):raise RuntimeError('Nonfinite/incompatible checkpoint preserved: '+rid)
                        if folder.exists() and not resume:
                            # Initialization produced no trained/committed state: archive, never delete.
                            archive=OUT/'initialization_archive'/f'{rid}_{time.time_ns()}';archive.parent.mkdir(parents=True,exist_ok=True)
                            folder.resolve().relative_to(OUT.resolve());archive.resolve().relative_to(OUT.resolve());folder.rename(archive)
                            if (CKPT/rid).exists():
                                dest=CKPT/'initialization_archive'/archive.name;dest.parent.mkdir(parents=True,exist_ok=True)
                                (CKPT/rid).resolve().relative_to(CKPT.resolve());dest.resolve().relative_to(CKPT.resolve());(CKPT/rid).rename(dest)
                            log.warning('Archived uncommitted initialization %s; no trained work restarted',rid)
                        stale_lock(log)
                        command=[sys.executable,'-u','-m','research.aggressive.train','--id',rid]+(['--resume'] if resume else [])+(['--fp32'] if s['fp32_recovery'] else [])
                        with (OUT/(rid+'.console.log')).open('ab') as file:proc=subprocess.Popen(command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                        s.update(status='running',pid=proc.pid,create_time=psutil.Process(proc.pid).create_time(),attempts=s['attempts']+1,resumed=resume,command=command)
                        state['status']='training';write_json(STATE,state);log.info('START %s PID %s resume=%s fp32_recovery=%s',rid,proc.pid,resume,s['fp32_recovery'])
                    while (proc.poll() is None if proc is not None else alive(s)):
                        if amended_boundary_handoff(s,rid,log):break
                        time.sleep(3)
                    if proc is not None:proc.wait()
                    if record.exists() and json.loads(record.read_text())['status']=='completed':
                        verify_saved(rid,c);record_stopping_audit(rid);s['status']='completed';write_json(STATE,state);log.info('COMPLETED/VERIFIED %s',rid);break
                    failure=folder/'failure_status.json';details=json.loads(failure.read_text()) if failure.exists() else {}
                    if planned_handoff_pending(rid) and s.get('pid')==json.loads((OUT/'stopping_policy_amendment.json').read_text()).get('legacy_worker_pid') and latest.exists() and compatible(spec,c):
                        s['status']='pending_closeout';write_json(STATE,state)
                        log.info('Planned meaningful-stop handoff; resume checkpoint for closeout, not a numerical retry');continue
                    if details.get('numerical') and details.get('precision')=='bf16' and not s['fp32_recovery'] and latest.exists() and compatible(spec,c):
                        s['fp32_recovery']=True
                        write_json(OUT/(rid+'_automatic_fp32_recovery.json'),dict(failure=details,policy='Predeclared one BF16-to-full-FP32 retry',resume_checkpoint=str(latest),checkpoint_sha256=sha256(latest)))
                        write_json(STATE,state);log.warning('Predeclared safe full-FP32 recovery %s',rid);continue
                    if not details and latest.exists() and compatible(spec,c) and s['attempts']<3:
                        log.warning('Recoverable process interruption; resume same stage %s',rid);continue
                    s['status']='failed';write_json(STATE,state);raise RuntimeError('Stage failed after safe precision policy; diagnostics preserved: '+rid)
            state['status']='postprocessing';write_json(STATE,state);log.info('Fixed weighted/equal results and validation-only Grad-CAM/CBAM/occlusion')
            from research.aggressive.results import closeout
            result=closeout(c);preserved();state.update(status='completed',primary_accuracy=result['accuracy'],primary_macro_f1=result['macro_f1'],no_more_performance_runs=True,old_test_used=False)
            write_json(STATE,state);log.info('COMPLETED: enhanced development study; STOP regardless of score');publish(c,log)
        except BaseException as exc:
            state.update(status='interrupted',error=repr(exc));write_json(STATE,state);log.exception('Queue stopped safely; rerun identical master for compatible recovery');raise

if __name__=='__main__':main()
