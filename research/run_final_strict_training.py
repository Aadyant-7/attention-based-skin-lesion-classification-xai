"""One authorized launch: three frozen strict trainings, fixed validation fusion, STOP."""
import argparse,json,logging,os,subprocess,sys,time
import psutil
import torch
from .common import ROOT,write_json,sha256
from .strict_protocol import OUT,verification
from .strict_train import run_lock,validate_config,development_data,code_hashes,runtime_versions,tensor_nonfinite_names
from .final_strict_results import verify_model,closeout
from .registry import read_registry,REGISTRY

QUEUE=ROOT/'research/configs/final_strict_queue_v1.json'
STATE=OUT/'status.json'

def alive(info,pid_key='pid'):
    try:
        p=psutil.Process(info[pid_key]);return p.is_running() and abs(p.create_time()-info['create_time'])<1
    except (psutil.NoSuchProcess,KeyError):return False

def compatible(config):
    path=ROOT/'checkpoints/structured'/config['experiment_id']/'latest.pt'
    k=torch.load(path,map_location='cpu',weights_only=False)
    return (k['config']==config and k['code_hashes']==code_hashes() and k['runtime_versions']==runtime_versions()
        and [r['epoch'] for r in k['history']]==list(range(1,len(k['history'])+1))
        and len(k['history'])<=50 and not tensor_nonfinite_names(k['model']) and not tensor_nonfinite_names(k['optimizer']))

def preserve_registry(state):
    current={r['experiment_id']:r for r in read_registry()}
    if any(current.get(r['experiment_id'])!=r for r in state['baseline_registry']):raise RuntimeError('Historical registry evidence changed')

def clear_stale_lock(log):
    lock=REGISTRY.with_suffix('.lock')
    if not lock.exists():return
    for p in psutil.process_iter(['cmdline','cwd']):
        try:
            if p.info['cwd'] and Path(p.info['cwd']).resolve()==ROOT and any('research.'+runner in ' '.join(p.info['cmdline'] or []) for runner in ['train','strict_train']):return
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    lock.unlink();log.warning('Removed orphaned registry lock; no training writer active')

from pathlib import Path

def publish(configs,log):
    paths=['research/FINAL_STRICT_VALIDATION_RESULTS.md','results/final_strict/v1','results/master_experiment_registry.csv']
    paths += ['results/structured_experiments/'+c['experiment_id'] for c in configs]
    try:
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='structured-research':raise RuntimeError('Branch changed; publication deferred')
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:raise RuntimeError('Unrelated staged changes; publication deferred')
        subprocess.run(['git','add','--']+paths,cwd=ROOT,check=True,timeout=60)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
            subprocess.run(['git','commit','-m','Close frozen strict-validation training and equal ensemble'],cwd=ROOT,check=True,timeout=60)
        subprocess.run(['git','push','origin','structured-research'],cwd=ROOT,check=True,timeout=60)
        write_json(OUT/'publication.json',dict(status='pushed',branch='structured-research'));log.info('Strict validation artifacts pushed')
    except Exception as e:
        write_json(OUT/'publication.json',dict(status='pending',reason=repr(e)));log.warning('Results preserved locally; publication deferred: %s',e)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');args=ap.parse_args()
    queue=json.loads(QUEUE.read_text());configs=[json.loads((ROOT/p).read_text()) for p in queue['configs']]
    expected=[('efficientnet_b0','EfficientNet_B0_Weights.IMAGENET1K_V1'),('convnext_tiny','ConvNeXt_Tiny_Weights.IMAGENET1K_V1'),('efficientnet_v2_s','EfficientNet_V2_S_Weights.IMAGENET1K_V1')]
    if [(c['model'],c['weights']) for c in configs]!=expected or queue['ensemble_weights']!=[1/3]*3 or queue['execution_mode']!='sequential':raise ValueError('Frozen queue altered')
    for c in configs:validate_config(c)
    report=verification()
    if args.check:
        for c in configs:
            train,val,w=development_data(c)
            assert len(train)==7009 and len(val)==1503 and torch.isfinite(w).all()
        print(json.dumps(dict(status='ready_authorized',split=report,execution_mode='sequential',cuda_used=False)));return
    preflight=json.loads((OUT/'resource_preflight_complete.json').read_text())
    if preflight['status']!='passed' or preflight['models']!=3 or preflight['execution_mode']!='sequential':raise RuntimeError('GPU preflight not passed')
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'master.log',encoding='utf-8'),logging.StreamHandler()]);log=logging.getLogger('strict_queue')
    with run_lock('final_strict_master_v1'):
        state=json.loads(STATE.read_text()) if STATE.exists() else dict(status='initialized',candidates={c['experiment_id']:dict(status='pending') for c in configs},baseline_registry=read_registry())
        if state.get('source_hashes') and state['source_hashes']!=code_hashes():raise RuntimeError('Recovery source changed; no silent migration')
        if state.get('queue_sha256') and state['queue_sha256']!=sha256(QUEUE):raise RuntimeError('Queue changed')
        config_hashes={p:sha256(ROOT/p) for p in queue['configs']}
        if state.get('config_hashes') and state['config_hashes']!=config_hashes:raise RuntimeError('Configs changed')
        preserve_registry(state)
        state.update(master_pid=os.getpid(),create_time=psutil.Process().create_time(),source_hashes=code_hashes(),queue_sha256=sha256(QUEUE),config_hashes=config_hashes,execution_mode='sequential')
        write_json(STATE,state)
        if state['status']=='completed':log.info('Strict validation already completed; no training or test evaluation');publish(configs,log);return
        try:
            for c,config_path in zip(configs,queue['configs']):
                rid=c['experiment_id'];s=state['candidates'][rid];folder=ROOT/'results/structured_experiments'/rid
                record=folder/'record.json';latest=ROOT/'checkpoints/structured'/rid/'latest.pt'
                if record.exists() and json.loads(record.read_text())['status']=='completed':
                    verify_model(c);s.update(status='completed');log.info('VERIFY/SKIP %s',rid);write_json(STATE,state);continue
                if s.get('status')=='failed':raise RuntimeError('Numerically failed candidate preserved; no silent retry: '+rid)
                proc=None
                if alive(s):log.info('Reattached existing %s PID %s',rid,s['pid'])
                else:
                    if record.exists() and json.loads(record.read_text())['status']=='failed':
                        # A normal trainer failure is not a power cut; require diagnosis.
                        raise RuntimeError('Recorded failure preserved; no automatic numerical retry: '+rid)
                    resume=latest.exists()
                    if resume and not compatible(c):raise RuntimeError('Incompatible/nonfinite checkpoint preserved: '+rid)
                    if folder.exists() and not resume:raise RuntimeError('Partial initialization without checkpoint preserved: '+rid)
                    clear_stale_lock(log)
                    console=OUT/(rid+'.console.log');handle=console.open('ab')
                    command=[sys.executable,'-u','-m','research.strict_train','--config',config_path]+(['--resume'] if resume else [])
                    try:proc=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    finally:handle.close()
                    s.update(status='running',pid=proc.pid,create_time=psutil.Process(proc.pid).create_time(),resumed=resume,command=command)
                    state['status']='training';write_json(STATE,state);log.info('START %s PID %s resume=%s',rid,proc.pid,resume)
                while (proc.poll() is None if proc is not None else alive(s)):time.sleep(3)
                if proc is not None:proc.wait()
                if record.exists() and json.loads(record.read_text())['status']=='completed':
                    verify_model(c);s.update(status='completed');log.info('COMPLETED/VERIFIED %s',rid);write_json(STATE,state)
                elif record.exists() and json.loads(record.read_text())['status']=='failed':
                    s['status']='failed';write_json(STATE,state);raise RuntimeError('Candidate failed; diagnostics preserved: '+rid)
                elif latest.exists() and compatible(c):
                    s['status']='interrupted';write_json(STATE,state);raise RuntimeError('Unexpected process interruption; rerun same queue to resume '+rid)
                else:raise RuntimeError('No valid committed checkpoint; preserved '+rid)
            preserve_registry(state);state['status']='postprocessing';write_json(STATE,state)
            log.info('STRICT VALIDATION POSTPROCESS: fixed equal fusion; no test loader')
            delta=closeout(configs,queue['ensemble_id']);preserve_registry(state)
            state.update(status='completed',strict_validation_completed=True,test_evaluated=False,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),summary=delta)
            write_json(STATE,state);log.info('COMPLETED: strict validation saved; STOP before locked test')
            publish(configs,log)
        except Exception as e:
            state.update(status='interrupted',error=repr(e));write_json(STATE,state);log.exception('Strict batch stopped safely; all evidence preserved');raise

if __name__=='__main__':main()
