"""One launch: two approved candidate trainings, bounded CPU analysis, permanent freeze.
No test loader, no strict training, no automatic architecture proposals.
"""
import argparse,json,logging,os,subprocess,sys,time,traceback
from pathlib import Path
import psutil
import torch
from .common import ROOT,sha256,write_json
from .train import code_hashes,runtime_versions,run_lock,validate_config,tensor_nonfinite_names
from .registry import read_registry
from .final_selection import BATCH,CONFIG,OUT,OLD,S12,REF,DENSE_FUSION,evidence,verify_training,postprocess

STATE=BATCH/'status.json'

def safe_resume(config):
 p=ROOT/'checkpoints/structured'/config['experiment_id']/'latest.pt'
 if not p.exists():return False
 k=torch.load(p,map_location='cpu',weights_only=False)
 return k['config']==config and k['code_hashes']==code_hashes() and k['runtime_versions']==runtime_versions() and len(k['history'])<=config['max_epochs'] and [r['epoch'] for r in k['history']]==list(range(1,len(k['history'])+1)) and not tensor_nonfinite_names(k['model']) and not tensor_nonfinite_names(k['optimizer'])

def recoverable(code,text):
 fatal=['FloatingPointError','Nonfinite','out of memory','ValueError','AssertionError','FileNotFoundError','Unapproved','Resume config or runner changed']
 if any(x.lower() in text.lower() for x in fatal):return False
 return code<0 or code in (130,3221225786) or any(x in text for x in ['KeyboardInterrupt','WinError 995','WinError 1450','temporarily unavailable'])

def process_alive(info):
 try:
  p=psutil.Process(info['pid'])
  return abs(p.create_time()-info['create_time'])<1 and p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
 except (psutil.NoSuchProcess,KeyError):return False

def unchanged(manifest):
 for name,digest in manifest['preserved_files'].items():
  if sha256(ROOT/name)!=digest:raise RuntimeError('Historical evidence changed: '+name)
 original={r['experiment_id']:r for r in manifest['baseline_registry']}
 current={r['experiment_id']:r for r in read_registry()}
 if any(current.get(r)!=v for r,v in original.items()):raise RuntimeError('Pre-batch registry row changed')

def audit(manifest):
 assert manifest['candidate_count']==2 and len(manifest['candidates'])==2
 for name,h in manifest['source_hashes'].items():assert sha256(ROOT/name)==h,'Batch source changed'
 for c in manifest['candidates']:
  config=json.loads((ROOT/c['config']).read_text());validate_config(config)
  assert config['experiment_id']==c['id'] and config['max_epochs']==20 and config['protocol']=='exploratory_image_level'
  assert sha256(ROOT/c['config'])==c['config_sha256']
 unchanged(manifest)
 for r in OLD+[S12,REF,DENSE_FUSION]:evidence(r)
 assert manifest['execution_mode'] in ['sequential','parallel_two']
 return True

def clear_stale_registry_lock(log):
    from .registry import REGISTRY
    lock=REGISTRY.with_suffix('.lock')
    if not lock.exists():return
    for process in psutil.process_iter(['cmdline','cwd']):
        try:
            cmd=' '.join(process.info['cmdline'] or [])
            if 'research.train' in cmd and process.info['cwd'] and Path(process.info['cwd']).resolve()==ROOT.resolve():return
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    lock.resolve().relative_to(ROOT)
    lock.unlink();log.warning('Removed stale registry lock after confirming no project training writer')


def publish_results(manifest,log):
    """Publish only this batch; never stage checkpoints or unrelated user edits."""
    paths=['research/FINAL_ARCHITECTURE_SELECTION.md','research/FINAL_HIGH_PERFORMANCE_RECIPE.md','research/FINAL_50_EPOCH_TRAINING_PLAN.md','research/MODEL_VS_ACCURACY.md',str(OUT.relative_to(ROOT)),str(BATCH.relative_to(ROOT))]
    paths += ['results/structured_experiments/'+c['id'] for c in manifest['candidates']]
    paths += [r['config_path'].rsplit('/',1)[0] for r in read_registry() if r['experiment_id'].startswith(tuple('s'+str(i)+'_final_' for i in range(20,28)))]
    paths += ['results/master_experiment_registry.csv']
    paths=[p for p in paths if (ROOT/p).exists()]
    try:
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
        if branch!='structured-research':raise RuntimeError('Branch changed; publication deferred')
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode!=0:raise RuntimeError('Unrelated staged changes exist; publication deferred')
        subprocess.run(['git','add','--']+paths,cwd=ROOT,check=True,timeout=60)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode!=0:
            subprocess.run(['git','commit','-m','Freeze final architecture selection and preserve bounded batch evidence'],cwd=ROOT,check=True,timeout=60)
        subprocess.run(['git','push','origin','structured-research'],cwd=ROOT,check=True,timeout=60)
        write_json(BATCH/'publication.json',dict(status='pushed',branch=branch));log.info('Final artifacts committed/pushed')
    except Exception as e:
        write_json(BATCH/'publication.json',dict(status='pending',reason=repr(e)));log.warning('Local freeze complete; publication deferred: %s',e)


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');args=ap.parse_args()
 manifest=json.loads(CONFIG.read_text(encoding='utf-8'));audit(manifest)
 if args.check:
  print(json.dumps(dict(status='ready',models=[c['id'] for c in manifest['candidates']],execution_mode=manifest['execution_mode'],automatic_postprocessing=True,test_loader=False,gpu_used=False)));return
 BATCH.mkdir(parents=True,exist_ok=True)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(BATCH/'master.log',encoding='utf-8'),logging.StreamHandler()]);log=logging.getLogger('final_batch')
 with run_lock('final_architecture_selection_master_v1'):
  state=json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else dict(status='initialized',candidates={c['id']:dict(status='pending',attempts=0) for c in manifest['candidates']})
  if state['status']=='completed':
   log.info('Architecture already frozen; skip all training.');publish_results(manifest,log);return
  state.update(status='training',master_pid=os.getpid(),create_time=psutil.Process().create_time(),manifest_sha256=sha256(CONFIG),execution_mode=manifest['execution_mode'])
  write_json(STATE,state)
  active={};handles={};slots=2 if manifest['execution_mode']=='parallel_two' else 1
  try:
   while True:
    # Reattach any orphaned candidate before allocating GPU slots; no duplicate run.
    for c in manifest['candidates']:
     rid=c['id'];s=state['candidates'][rid]
     if rid not in active and s['status']=='running' and process_alive(s):active[rid]=None;log.info('Reattached live child %s PID %s',rid,s['pid'])
    for c in manifest['candidates']:
     rid=c['id'];s=state['candidates'][rid];p=ROOT/'results/structured_experiments'/rid
     if rid in active or s['status'] in ['completed','failed']:continue
     record=p/'record.json'
     if record.exists() and json.loads(record.read_text())['status']=='completed':
      try:verify_training(rid);s.update(status='completed',reason='existing completed candidate verified/skipped');log.info('Verified and skipped %s',rid)
      except Exception as e:s.update(status='failed',reason='Artifact verification failed: '+repr(e));log.exception('Verification failure %s',rid)
      write_json(STATE,state);continue
     if len(active)>=slots:continue
     if record.exists() and json.loads(record.read_text())['status']=='failed':
      console=BATCH/(rid+'.console.log');text=console.read_text(encoding='utf-8',errors='replace')[-12000:] if console.exists() else ''
      if not recoverable(-1,text) or any(word.lower() in text.lower() for word in ['FloatingPointError','Nonfinite','out of memory','ValueError','AssertionError','FileNotFoundError']):
       s.update(status='failed',reason='Unrecoverable prior candidate failure preserved: '+text[-1800:]);write_json(STATE,state);continue
     if not active:clear_stale_registry_lock(log)
     config=json.loads((ROOT/c['config']).read_text());resume=(ROOT/'checkpoints/structured'/rid/'latest.pt').exists()
     if resume and not safe_resume(config):s.update(status='failed',reason='Checkpoint/source/runtime mismatch; preserved, no automatic migration');write_json(STATE,state);continue
     if p.exists() and not resume:s.update(status='failed',reason='Partial initialization without committed checkpoint; preserved, cannot safely resume');write_json(STATE,state);continue
     # On master restart a still-running child was reattached; a terminated valid child resumes here.
     console=BATCH/(rid+'.console.log');h=console.open('ab');cmd=[sys.executable,'-u','-m','research.train','--config',c['config']]+(['--resume'] if resume else [])
     proc=subprocess.Popen(cmd,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
     handles[rid]=h;active[rid]=proc;s.update(status='running',pid=proc.pid,create_time=psutil.Process(proc.pid).create_time(),attempts=s.get('attempts',0)+1,command=cmd,resumed=resume,console=str(console.relative_to(ROOT)))
     write_json(STATE,state);log.info('START %s PID %s resume=%s',rid,proc.pid,resume)
    for rid,proc in list(active.items()):
     s=state['candidates'][rid]
     if (proc is not None and proc.poll() is None) or (proc is None and process_alive(s)):continue
     code=proc.returncode if proc is not None else -1
     if rid in handles:handles.pop(rid).close()
     del active[rid]
     c=next(c for c in manifest['candidates'] if c['id']==rid);p=ROOT/'results/structured_experiments'/rid;record=p/'record.json'
     if record.exists() and json.loads(record.read_text())['status']=='completed':
      try:verify_training(rid);s.update(status='completed',exit_code=code);log.info('COMPLETED and verified %s',rid)
      except Exception as e:s.update(status='failed',reason='Verification failure '+repr(e));log.exception('Invalid candidate artifact %s',rid)
     else:
      text=(BATCH/(rid+'.console.log')).read_text(encoding='utf-8',errors='replace')[-12000:]
      config=json.loads((ROOT/c['config']).read_text())
      if s['attempts']<2 and recoverable(code,text) and safe_resume(config):s.update(status='pending',reason='One safe automatic interruption resume');log.warning('Recoverable interruption %s; resume scheduled',rid)
      else:s.update(status='failed',exit_code=code,reason=text[-2500:] or 'No completed artifact');log.error('FAILED %s; preserved; continuing next candidate',rid)
     write_json(STATE,state)
    if all(s['status'] in ['completed','failed'] for s in state['candidates'].values()) and not active:break
    time.sleep(3)
   unchanged(manifest);state['status']='postprocessing';write_json(STATE,state);log.info('AUTO POSTPROCESS: comparisons, complementarity, bounded ensembles, architecture freeze, future strict recipe')
   frozen=postprocess(manifest,state);unchanged(manifest)
   state.update(status='completed',architecture_selection_closed=True,selected_run=frozen['selected_run'],finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
   write_json(STATE,state);log.info('FINAL ARCHITECTURE FROZEN %s; no more training or architectures',frozen['selected_run']);publish_results(manifest,log)
  except Exception as e:
   state.update(status='batch_interrupted' if active else 'postprocessing_failed',error=repr(e));write_json(STATE,state);log.exception('Batch interrupted; rerun same master command safely');raise
  finally:
   for h in handles.values():h.close()

if __name__=='__main__':main()
