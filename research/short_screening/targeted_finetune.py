"""Paired, capped ConvNeXt sampling intervention. Development data only."""
import argparse,gc,json,logging,os,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil,torch
from torch.utils.data import DataLoader
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv,atomic_text
from research.models import ResearchClassifier
from research.strict_protocol import development_data,partition_metadata
from research.strict_train import DevelopmentImages,optimizer_groups,weighted_numerator,metric_report,atomic_checkpoint,rng_state,restore_rng,runtime_versions,run_lock,seed_worker,tensor_nonfinite_names
from research.plots import metric_figures,training_figures,comparison_figures
from research.aggressive.core import retry_registry_upsert,weighted_metrics,predictions
from research.registry import FIELDS

OUT=ROOT/'results/short_screening/targeted_finetune_v1'
CK=ROOT/'checkpoints/short_screening/targeted_finetune_v1'
CFG=ROOT/'research/short_screening/targeted_finetune_v1.json'
SOURCE=ROOT/'checkpoints/structured/s29_convnext_tiny_final_strict_seed42/best.pt'
SOURCE_SHA='ffe3a7f00c5371824fa46a8f9e5f864f53e15fcc533f5bf0263648f8573a1e93'
IDS=dict(control='s37_convnext_control_strict_dev_seed42',targeted='s38_convnext_melbkl_sampling_strict_dev_seed42')

def recipe():return json.loads(CFG.read_text())

def fingerprints():
    names=['research/short_screening/targeted_finetune.py','research/short_screening/targeted_finetune_v1.json',
        'research/strict_train.py','research/strict_protocol.py','research/models.py','research/common.py','research/plots.py','research/registry.py','research/aggressive/core.py']
    return {n:sha256(ROOT/n) for n in names}

def batches(labels,epoch,targeted):
    order=torch.arange(len(labels))
    focus=torch.tensor(np.flatnonzero(np.isin(labels,[2,4])),dtype=torch.long)
    extra=focus if targeted else torch.randperm(len(labels),generator=torch.Generator().manual_seed(4242+epoch))[:len(focus)]
    order=torch.cat([order,extra])
    order=order[torch.randperm(len(order),generator=torch.Generator().manual_seed(42+epoch))]
    order=order[:len(order)//32*32]
    return [part.tolist() for part in order.split(32)]

def metrics_from_saved(c,val):
    frames=[]
    for rid in ['s28_efficientnet_b0_final_strict_seed42','s29_convnext_tiny_final_strict_seed42','s30_efficientnet_v2_s_final_strict_seed42']:
        p=ROOT/'results/structured_experiments'/rid/'validation_predictions.csv';frame=pd.read_csv(p)
        assert not frame.image_id.duplicated().any() and set(frame.image_id)==set(val.image_id)
        frame=frame.set_index('image_id').loc[val.image_id].reset_index()
        assert frame.true_class.tolist()==val.diagnosis.tolist()
        array=frame[[f'p_{cl}' for cl in CLASSES]].to_numpy()
        assert np.isfinite(array).all() and np.allclose(array.sum(1),1,atol=1e-5)
        frames.append(array)
    return frames

def report(y,p,loss):return weighted_metrics(np.array(y),np.array(p),loss)

def ensemble(chosen,val,arrays):
    p=pd.DataFrame(chosen['predictions']).set_index('image_id').loc[val.image_id][[f'p_{cl}' for cl in CLASSES]].to_numpy()
    fused=(arrays[0]+p+arrays[2])/3
    loss=float(-np.log(np.maximum(fused[np.arange(len(val)),val.label],1e-12)).mean())
    return report(val.label,fused,loss),predictions(val.image_id,val.label,fused)

def gate(candidate,control,baseline):
    # Decisions use the fixed ensemble of each arm's standalone accuracy winner.
    targets=[control,baseline]
    checks=dict(accuracy=all(candidate['accuracy']-m['accuracy']>=.005-1e-12 for m in targets),
        macro_f1=all(candidate['macro_f1']>=m['macro_f1'] for m in targets),
        focus_recall=all(np.mean([candidate['per_class'][cl]['recall']-m['per_class'][cl]['recall'] for cl in ['mel','bkl']])>=.02 for m in targets),
        neither_focus_class_declines=all(candidate['per_class'][cl]['recall']>=m['per_class'][cl]['recall']-.01 for m in targets for cl in ['mel','bkl']),
        nv_recall=all(candidate['per_class']['nv']['recall']>=m['per_class']['nv']['recall']-.01 for m in targets))
    return dict(passed=all(checks.values()),checks=checks)

def registry(arm,c,history,status,best=None):
    folder=OUT/IDS[arm];row={k:'' for k in FIELDS}
    row.update(experiment_id=IDS[arm],era='structured',record_kind='warm_start_finetuning',phase='short_screening',protocol='post_test_strict_development',evaluation_split='validation',
        model='convnext_tiny',pretrained_weights=c['weights'],attention='none',method=c['question'],seed=42,image_size=224,
        preprocessing=c['preprocessing'],augmentation=c['augmentation'],imbalance='sqrt CE weights; '+('MEL/BKL repeated once per epoch' if arm=='targeted' else 'uniform extra exposure with same draw/update budget'),
        loss='weighted_cross_entropy',optimizer='AdamW',backbone_lr=c['learning_rate']['backbone'],head_lr=c['learning_rate']['head'],batch_size=16,
        epochs=len(history),status=status,config_path=relative(CFG),history_path=relative(folder/'history.csv'),source_sha256=SOURCE_SHA,
        notes='New post-test development intervention; warm start from S29 accuracy epoch33; no original method/test changes; checkpoint epoch0 means retained source.')
    if best:row.update(best_epoch=best['epoch'],checkpoint=relative(CK/IDS[arm]/'best.pt'),checkpoint_available_local=True,
        metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),**{k:best['metrics'][k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']})
    return row

def validate(model,loader,w):
    model.eval();y=[];p=[];ids=[];numerator=denominator=0.
    with torch.inference_mode():
        for x,t,image_ids in loader:
            x=x.cuda(non_blocking=True);t=t.cuda(non_blocking=True);z=model(x.float());n=weighted_numerator(z,t,w);a=z.float().softmax(1)
            if not torch.isfinite(z).all() or not torch.isfinite(n) or not torch.isfinite(a).all():raise FloatingPointError('Nonfinite FP32 validation; no invalid output accepted')
            numerator+=float(n);denominator+=float(w[t].sum());y.extend(t.cpu().tolist());p.extend(a.cpu().tolist());ids.extend(image_ids)
    return report(y,p,numerator/denominator),predictions(ids,y,p)

def worker(arm,resume):
    c=recipe();rid=IDS[arm];folder=OUT/rid;ck=CK/rid
    if (folder/'record.json').exists() and json.loads((folder/'record.json').read_text()).get('status')=='completed':return
    assert sha256(SOURCE)==SOURCE_SHA and c['maximum_total_gpu_epochs']==20
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    train,val,w=development_data(c);arrays=metrics_from_saved(c,val);w=w.cuda()
    baseline=json.loads((ROOT/'results/final_strict/v1/ensemble/validation_metrics.json').read_text())
    if not resume:folder.mkdir(parents=True,exist_ok=False);ck.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(folder/'train.log',encoding='utf-8'),logging.StreamHandler()]);log=logging.getLogger(rid)
    model=ResearchClassifier('convnext_tiny',None,'none',.2).cuda()
    optimizer=torch.optim.AdamW(optimizer_groups(model,c),betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,threshold=.002,threshold_mode='abs',min_lr=1e-7)
    scaler=torch.amp.GradScaler('cuda');history=[];best=f1best=None;runtime=0.;updates=0;reason='hard_epoch_cap'
    if resume:
        snap=torch.load(ck/'latest.pt',map_location='cpu',weights_only=False)
        assert snap['config']==c and snap['arm']==arm and snap['fingerprints']==fingerprints() and snap['runtime_versions']==runtime_versions()
        if tensor_nonfinite_names(snap['model']) or tensor_nonfinite_names(snap['optimizer']):raise FloatingPointError('Invalid resume state preserved')
        model.load_state_dict(snap['model']);optimizer.load_state_dict(snap['optimizer']);scheduler.load_state_dict(snap['scheduler']);scaler.load_state_dict(snap['scaler'])
        history=snap['history'];best=snap['best'];f1best=snap['f1best'];runtime=snap['runtime_seconds'];updates=snap['updates'];restore_rng(snap['rng']);del snap;gc.collect()
    else:
        parent=torch.load(SOURCE,map_location='cpu',weights_only=False);assert parent['best_epoch']==33 and parent['class_order']==list(CLASSES)
        model.load_state_dict(parent['model'],strict=True)
        frame=pd.read_csv(ROOT/'results/structured_experiments/s29_convnext_tiny_final_strict_seed42/validation_predictions.csv').set_index('image_id').loc[val.image_id].reset_index()
        best=dict(epoch=0,metrics=parent['metrics'],predictions=frame.to_dict('records'),model=parent['model']);f1best=best;del parent
        write_json(folder/'initialization.json',dict(warm_start=relative(SOURCE),sha256=SOURCE_SHA,source_epoch=33,new_optimizer=True,arm=arm,recipe=c,source_fingerprints=fingerprints()))
    tick=time.perf_counter()
    def payload():return dict(config=c,arm=arm,fingerprints=fingerprints(),runtime_versions=runtime_versions(),model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
        history=history,best=best,f1best=f1best,rng=rng_state(True),runtime_seconds=runtime+time.perf_counter()-tick,updates=updates,selection='earliest raw maximum accuracy; source retained at epoch0 if no improvement')
    def checkpoint_best(name,chosen):atomic_checkpoint(ck/name,dict(config=c,arm=arm,model=chosen['model'],best_epoch=chosen['epoch'],metrics=chosen['metrics'],fingerprints=fingerprints(),source_sha256=SOURCE_SHA))
    if not resume:atomic_checkpoint(ck/'latest.pt',payload());checkpoint_best('best.pt',best);checkpoint_best('best_macro_f1.pt',f1best)
    val_loader=DataLoader(DevelopmentImages(val,c),batch_size=16,shuffle=False,num_workers=2,pin_memory=True)
    cap=5 if arm=='control' else 15
    log.info('START %s resume=%s source_epoch33 train=%d val=%d cap=%d; no test loader',rid,resume,len(train),len(val),cap)
    retry_registry_upsert(registry(arm,c,history,'running'))
    try:
        for epoch in range(len(history)+1,cap+1):
            if arm=='targeted' and len(history) in [5,10]:
                selected_ensemble,_=ensemble(best,val,arrays);control=json.loads((OUT/IDS['control']/'ensemble_metrics.json').read_text())
                decision=gate(selected_ensemble,control,baseline)
                write_json(folder/'continuation_gate.json',dict(at_completed_epoch=len(history),**decision))
                if not decision['passed']:reason='pilot_material_gate_failed';break
                if len(history)>=10:
                    e5=json.loads((folder/'screening_epoch5.json').read_text())['selected_ensemble_metrics']
                    if selected_ensemble['accuracy']<e5['accuracy']+.002:reason='no_material_progress_after_pilot';break
            epoch_tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();model.train();sampler=batches(train.label.to_numpy(),epoch,arm=='targeted')
            loader=DataLoader(DevelopmentImages(train,c,True),batch_sampler=sampler,num_workers=2,pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42+epoch))
            num_total=den_total=0.;correct=used=skips=0;lr_before=[g['lr'] for g in optimizer.param_groups]
            for step,(x,t,image_ids) in enumerate(loader):
                x=x.cuda(non_blocking=True);t=t.cuda(non_blocking=True);den=float(w[t].sum());optimizer.zero_grad(set_to_none=True)
                for start in [0,16]:
                    with torch.autocast('cuda',dtype=torch.float16):z=model(x[start:start+16])
                    n=weighted_numerator(z,t[start:start+16],w)
                    if not torch.isfinite(z).all() or not torch.isfinite(n):raise FloatingPointError('Nonfinite training forward; preserve checkpoint and diagnostics')
                    scaler.scale(n/den).backward();num_total+=float(n.detach());correct+=int((z.argmax(1)==t[start:start+16]).sum())
                scaler.unscale_(optimizer)
                finite=bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all())
                if finite:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();skipped=int(scaler.get_scale()<old_scale);skips+=skipped;updates+=1-skipped
                if skipped:log.warning('GradScaler rejected a scaled-gradient overflow; no sample removed')
                den_total+=den;used+=len(t)
                if step%25==0:
                    write_json(folder/'progress.json',dict(epoch=epoch,step=step+1,steps=len(sampler),cap=cap,arm=arm))
                    log.info('epoch %d/%d step %d/%d weighted_loss=%.5f',epoch,cap,step+1,len(sampler),num_total/den_total)
            m,pred=validate(model,val_loader,w);scheduler.step(m['accuracy'])
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if m['accuracy']>best['metrics']['accuracy']:best=dict(epoch=epoch,metrics=m,predictions=pred,model=state);checkpoint_best('best.pt',best)
            if m['macro_f1']>f1best['metrics']['macro_f1']:f1best=dict(epoch=epoch,metrics=m,predictions=pred,model=state);checkpoint_best('best_macro_f1.pt',f1best)
            history.append(dict(epoch=epoch,train_loss=num_total/den_total,train_accuracy=correct/used,train_images_used=used,unique_train_images_used=len(set(i for b in sampler for i in b)),
                val_loss=m['loss'],val_accuracy=m['accuracy'],val_macro_precision=m['macro_precision'],val_macro_recall=m['macro_recall'],val_macro_f1=m['macro_f1'],
                backbone_lr=lr_before[0],head_lr=lr_before[1],backbone_lr_after=optimizer.param_groups[0]['lr'],head_lr_after=optimizer.param_groups[1]['lr'],
                amp_skipped_updates=skips,epoch_seconds=time.perf_counter()-epoch_tick,peak_allocated_vram_mb=torch.cuda.max_memory_allocated()/2**20))
            atomic_checkpoint(ck/'latest.pt',payload());write_csv(folder/'history.csv',history)
            write_json(folder/'validation_metrics_latest.json',m);write_csv(folder/'validation_predictions_latest.csv',pred)
            if epoch==5:
                em,_=ensemble(best,val,arrays);write_json(folder/'screening_epoch5.json',dict(selected_ensemble_metrics=em,best_epoch=best['epoch'],best_metrics=best['metrics']))
            retry_registry_upsert(registry(arm,c,history,'running',best))
            log.info('COMMITTED epoch %d accuracy=%.6f macroF1=%.6f best_epoch=%d',epoch,m['accuracy'],m['macro_f1'],best['epoch']);del state
        em,ep=ensemble(best,val,arrays)
        for suffix,chosen in [('',best),('_macro_f1',f1best)]:
            write_json(folder/f'validation_metrics{suffix}.json',chosen['metrics']);write_csv(folder/f'validation_predictions{suffix}.csv',chosen['predictions'])
            write_csv(folder/f'validation_probabilities{suffix}.csv',pd.DataFrame(chosen['predictions'])[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
            metric_figures(chosen['metrics'],folder/('figures'+suffix),rid+' | post-test strict development')
        write_json(folder/'ensemble_metrics.json',em);write_csv(folder/'ensemble_predictions.csv',ep);metric_figures(em,folder/'ensemble_figures',rid+' | fixed B0 + candidate + V2S')
        training_figures(pd.DataFrame(history),folder/'figures',rid+' | warm-start intervention',selection_metric='accuracy')
        write_csv(folder/'lr_history.csv',[{key:r[key] for key in ['epoch','backbone_lr','head_lr','backbone_lr_after','head_lr_after']} for r in history])
        summary=dict(status='completed',new_epochs=len(history),hard_cap=cap,best_epoch=best['epoch'],best_macro_f1_epoch=f1best['epoch'],stop_reason=reason,runtime_seconds=runtime+time.perf_counter()-tick,
            optimizer_updates=updates,baseline_retained=best['epoch']==0,source_epoch=33,test_loaded=False)
        write_json(folder/'summary.json',summary);row=registry(arm,c,history,'completed',best);row['checkpoint_sha256']=sha256(ck/'best.pt');row['decision']=reason
        retry_registry_upsert(row);write_json(folder/'record.json',row);log.info('COMPLETED %s reason=%s new_epochs=%d',rid,reason,len(history))
    except BaseException as exc:
        stamp='failure_'+str(time.time_ns());write_json(folder/(stamp+'.json'),dict(error=repr(exc),committed_epochs=len(history)))
        atomic_checkpoint(folder/(stamp+'.pt'),dict(diagnostic_only=True,**payload()));log.exception('Stopped; committed checkpoints preserved');raise

def prepare():
    c=recipe();assert sha256(SOURCE)==SOURCE_SHA;train,val,w=development_data(c)
    report=partition_metadata()[1];write_json(OUT/'split_verification.json',report)
    rows=[]
    for arm in IDS:
        indices=np.array([i for b in batches(train.label.to_numpy(),1,arm=='targeted') for i in b])
        rows.append(dict(arm=arm,epoch_samples=len(indices),unique_images=len(set(indices)),class_exposure={cl:int((train.label.to_numpy()[indices]==j).sum()) for j,cl in enumerate(CLASSES)}))
    write_json(OUT/'sampling_preflight.json',dict(arms=rows,weights=w.tolist(),source_checkpoint=relative(SOURCE),source_sha256=SOURCE_SHA,
        model='convnext_tiny',warm_start_epoch=33,max_total_epochs=20,source_fingerprints=fingerprints(),cpu_passed=True))
    # Disposable fit probe only; never validation inference or an experiment epoch.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(4);torch.manual_seed(42)
    model=ResearchClassifier('convnext_tiny',None,'none',.2).cuda();snap=torch.load(SOURCE,map_location='cpu',weights_only=False);model.load_state_dict(snap['model']);del snap
    loader=DataLoader(DevelopmentImages(train,c,True),batch_sampler=batches(train.label.to_numpy(),1,True),num_workers=0)
    x,y,_=next(iter(loader));x=x.cuda();y=y.cuda();w=w.cuda();opt=torch.optim.AdamW(optimizer_groups(model,c),weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda');den=float(w[y].sum());model.train();torch.cuda.reset_peak_memory_stats()
    for start in [0,16]:
        with torch.autocast('cuda',dtype=torch.float16):z=model(x[start:start+16])
        n=weighted_numerator(z,y[start:start+16],w);assert torch.isfinite(n) and torch.isfinite(z).all();scaler.scale(n/den).backward()
    scaler.unscale_(opt);gradients_finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # A first FP16 loss-scale overflow is safely skipped; source/model must stay finite.
    scaler.step(opt);scaler.update();assert not tensor_nonfinite_names(model.state_dict()) and not tensor_nonfinite_names(opt.state_dict())
    write_json(OUT/'gpu_preflight.json',dict(device=torch.cuda.get_device_name(),peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
        logits_and_loss_finite=True,scaled_gradients_finite=bool(gradients_finite),probe_discarded=True,validation_accuracy_not_measured=True,source_sha256=SOURCE_SHA))
    print('Preflight passed; discarded probe; source checkpoint unchanged',flush=True)

def batch():
    OUT.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'master.log',encoding='utf-8'),logging.StreamHandler()]);log=logging.getLogger('targeted_queue')
    with run_lock('targeted_finetune_v1_master'):
        if not (OUT/'gpu_preflight.json').exists():raise RuntimeError('Prepared fit check required')
        state_path=OUT/'status.json'
        status=json.loads(state_path.read_text()) if state_path.exists() else dict(status='initialized',hard_total_epoch_cap=20,fingerprints=fingerprints())
        if status['fingerprints']!=fingerprints():raise RuntimeError('Code/config changed; no silent recovery')
        status.update(master_pid=os.getpid(),master_create_time=psutil.Process().create_time(),status='running')
        write_json(OUT/'status.json',status)
        for arm,rid in IDS.items():
            folder=OUT/rid;record=folder/'record.json'
            if record.exists() and json.loads(record.read_text()).get('status')=='completed':
                saved=json.loads(record.read_text())
                if saved['checkpoint_sha256']!=sha256(CK/rid/'best.pt'):raise RuntimeError('Completed checkpoint changed')
                log.info('SKIP completed %s',rid);continue
            latest=CK/rid/'latest.pt'
            command=[sys.executable,'-u','-m','research.short_screening.targeted_finetune','--arm',arm]+(['--resume'] if latest.exists() else [])
            attached=None
            if status.get('active_arm')==arm and psutil.pid_exists(status.get('worker_pid',-1)):
                possible=psutil.Process(status['worker_pid'])
                if abs(possible.create_time()-status.get('worker_create_time',0))<.01:attached=possible
            if attached:
                log.info('Reattached %s PID %d',rid,attached.pid)
                while attached.is_running() and abs(attached.create_time()-status['worker_create_time'])<.01:
                    time.sleep(3)
            else:
                with (OUT/(rid+'.console.log')).open('ab') as output:proc=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                status.update(active_arm=arm,worker_pid=proc.pid,worker_create_time=psutil.Process(proc.pid).create_time(),status='training');write_json(OUT/'status.json',status);log.info('START %s PID %d resume=%s',rid,proc.pid,latest.exists())
                exit_code=proc.wait()
                if exit_code!=0:status.update(status='interrupted');write_json(OUT/'status.json',status);raise RuntimeError('Worker stopped safely; inspect saved failure evidence')
            if not record.exists() or json.loads(record.read_text()).get('status')!='completed':
                status.update(status='interrupted');write_json(OUT/'status.json',status);raise RuntimeError('Worker exited without valid closeout; preserve evidence')
            log.info('COMPLETED %s',rid)
        rows=[]
        for arm,rid in IDS.items():
            m=json.loads((OUT/rid/'validation_metrics.json').read_text());em=json.loads((OUT/rid/'ensemble_metrics.json').read_text());summary=json.loads((OUT/rid/'summary.json').read_text())
            rows.extend([dict(display_name=arm+' standalone',accuracy=m['accuracy'],macro_f1=m['macro_f1']),dict(display_name=arm+' fixed ensemble',accuracy=em['accuracy'],macro_f1=em['macro_f1'])])
        comparison_figures(rows,OUT/'figures_comparison','Paired sampling intervention | post-test development');write_csv(OUT/'comparison.csv',rows)
        status.update(status='completed',stop=True,no_additional_experiments=True);write_json(OUT/'status.json',status);log.info('Bounded paired intervention complete; STOP')
        try:
            paths=['research/short_screening','results/short_screening','results/master_experiment_registry.csv']
            if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='structured-research':raise RuntimeError('Branch changed')
            if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:raise RuntimeError('Unrelated staged changes')
            subprocess.run(['git','add','--']+paths,cwd=ROOT,check=True)
            if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:subprocess.run(['git','commit','-m','Close bounded paired ConvNeXt sampling intervention'],cwd=ROOT,check=True)
            subprocess.run(['git','push','origin','structured-research'],cwd=ROOT,check=True,timeout=60)
        except Exception as exc:write_json(OUT/'publication_pending.json',dict(reason=repr(exc)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--batch',action='store_true');parser.add_argument('--arm',choices=list(IDS));parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.batch:batch()
    elif args.arm:
        with run_lock(IDS[args.arm]):worker(args.arm,args.resume)
    else:parser.error('Choose --prepare, --batch, or --arm')

if __name__=='__main__':main()
