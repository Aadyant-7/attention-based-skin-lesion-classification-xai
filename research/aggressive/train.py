"""Fresh enhanced training with atomic epoch recovery and predeclared BF16->FP32 fallback."""
import argparse,copy,gc,json,logging,os,random,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models
from research.common import ROOT,write_json,write_csv,sha256,relative
from research.train import atomic_checkpoint,rng_state,restore_rng,runtime_versions,run_lock,seed_worker,tensor_nonfinite_names
from research.registry import FIELDS,upsert
from research.plots import metric_figures,training_figures
from .core import config,CONFIG,OUT,CKPT,hashes,data,Images,EnhancedModel,mixed_focal_sum,weighted_metrics,predictions,stopping_state,epoch_batches,micro_ranges
from .core import compatible_sources
from .stopping import stopping_status

class NumericalFailure(FloatingPointError):pass

def registry_row(spec,c,history,status):
    row={k:'' for k in FIELDS};rid=spec['id']
    row.update(experiment_id=rid,era='structured',record_kind='training_run',phase='aggressive_enhanced_post_test',protocol='enhanced_lesion_disjoint_development',
        evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],method='Predeclared complete literature-inspired post-test development recipe',
        model=spec['model'],pretrained_weights=spec['weights'],attention=spec['attention'],status=status,config_path=relative(OUT/rid/'config.json'),
        image_size=224,preprocessing='RGB224 bilinear; TRAIN-only mean/std',augmentation='class-specific spatial/color/erasing; Mixup .3/alpha .2',
        imbalance='training-only bounded inverse-frequency weights; no resampling',loss='FP32 weighted focal gamma2.2; correctly blended Mixup focal',optimizer='AdamW',
        backbone_lr=spec['lr'],head_lr=spec['lr'],batch_size=c['microbatch'],seed=c['seed'],epochs=len(history),history_path=relative(OUT/rid/'history.csv'),
        notes='Started after test revealed; excludes old test; not an untouched-test result. Original S31 held-out result preserved.')
    return row

def evaluate(model,loader,w,c):
    model.eval();ys=[];probs=[];ids=[];num=den=0.
    with torch.inference_mode():
        for x,y,image_ids in loader:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            logits=model(x.float())
            if not torch.isfinite(logits).all():raise NumericalFailure('Nonfinite FP32 validation logits')
            value=mixed_focal_sum(logits,y,y,1.,w,c['focal_gamma'])
            p=logits.float().softmax(1)
            if not torch.isfinite(value) or not torch.isfinite(p).all():raise NumericalFailure('Nonfinite FP32 validation loss/probability')
            num+=float(value);den+=float(w[y].sum());ys.extend(y.cpu().tolist());probs.extend(p.cpu().tolist());ids.extend(image_ids)
    return weighted_metrics(ys,probs,num/den),predictions(ids,ys,probs)

def run(spec,resume=False,force_fp32=False):
    c=config();rid=spec['id'];folder=OUT/rid;ck=CKPT/rid
    if (folder/'record.json').exists() and json.loads((folder/'record.json').read_text())['status']=='completed':return
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    train,val,w=data(c);snapshot=None
    if resume:
        snapshot=torch.load(ck/'latest.pt',map_location='cpu',weights_only=False)
        if snapshot['config']!=c or snapshot['spec']!=spec or not compatible_sources(snapshot['source_hashes']) or snapshot['runtime_versions']!=runtime_versions():raise ValueError('Incompatible recovery checkpoint')
        if tensor_nonfinite_names(snapshot['model']) or tensor_nonfinite_names(snapshot['optimizer']):raise ValueError('Nonfinite resume state; preserved')
    else:
        folder.mkdir(parents=True,exist_ok=False);ck.mkdir(parents=True,exist_ok=False)
        write_json(folder/'config.json',dict(recipe=c,model_spec=spec))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(folder/'train.log',encoding='utf-8'),logging.StreamHandler()]);log=logging.getLogger(rid)
    model=EnhancedModel(spec,pretrained=not resume).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=spec['lr'],betas=(.9,.999),eps=1e-8,weight_decay=.0015)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=.7)
    scaler=torch.amp.GradScaler('cuda',enabled=False) # BF16/FP32 need no loss scaling.
    precision='fp32' if force_fp32 else spec.get('initial_precision',c['training_precision']);history=[];best=None;f1best=None;reference=-1.;stale=0;runtime=0.;updates=0
    if snapshot:
        model.load_state_dict(snapshot['model']);opt.load_state_dict(snapshot['optimizer']);scheduler.load_state_dict(snapshot['scheduler'])
        history=snapshot['history'];best=snapshot['best'];f1best=snapshot['f1best'];reference=snapshot['early_reference'];stale=snapshot['stale'];runtime=snapshot['runtime_seconds'];updates=snapshot['optimizer_updates']
        precision='fp32' if force_fp32 else snapshot['precision'];restore_rng(snapshot['rng'])
        write_csv(folder/'history.csv',history);del snapshot;gc.collect()
    if precision not in ['bf16','fp32'] or (precision=='bf16' and not torch.cuda.is_bf16_supported()):raise ValueError('Unsupported predeclared precision')
    if force_fp32:write_json(folder/'precision_amendment.json',dict(kind='predeclared_bf16_to_full_fp32',reason='Numerical failure evidence preserved; resume last valid committed boundary',training_precision='fp32',validation_precision='fp32',no_hyperparameter_change=True))
    w=w.cuda();val_loader=DataLoader(Images(val,c),batch_size=c['microbatch'],num_workers=c['workers'],shuffle=False,pin_memory=True)
    tick=time.perf_counter()
    def payload():return dict(config=c,spec=spec,source_hashes=hashes(),runtime_versions=runtime_versions(),model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
        history=history,best=best,f1best=f1best,early_reference=reference,stale=stale,meaningful_stopping=stopping_status(history),runtime_seconds=runtime+time.perf_counter()-tick,optimizer_updates=updates,precision=precision,rng=rng_state(True))
    if not resume:
        enum=models.get_model_weights(spec['model'])[spec['weights'].split('.')[-1]]
        cached=torch.hub.get_dir()+'/checkpoints/'+enum.url.rsplit('/',1)[-1]
        write_json(folder/'pretraining.json',dict(weights=spec['weights'],weights_sha256=sha256(cached),fresh_imagenet=True,no_previous_training_checkpoint=True,parameters=sum(p.numel() for p in model.parameters()),runtime_versions=runtime_versions()))
        atomic_checkpoint(ck/'latest.pt',payload())
    upsert(registry_row(spec,c,history,'running'))
    log.info('START %s resume=%s precision=%s validation=FP32 train=%d val=%d no test loader',rid,resume,precision,len(train),len(val))
    log.info('Meaningful stopping v2: accuracy +.002 OR macroF1 +.003; min25/max50; patience12; plateau10 after30; LR settle3')
    try:
        for epoch in range(len(history)+1,51):
            stopping=stopping_status(history)
            if stopping['stop']:
                atomic_checkpoint(ck/'latest.pt',payload())
                log.info('STOP at committed boundary: %s',stopping['reason']);break
            epoch_tick=time.perf_counter();model.train();stage=model.stage(epoch);torch.cuda.reset_peak_memory_stats()
            loader=DataLoader(Images(train,c,True),batch_sampler=epoch_batches(len(train),c['seed']+epoch),num_workers=c['workers'],pin_memory=True,
                worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(c['seed']+epoch))
            numerator=denominator=0.;correct=used=0.;mixed_count=0
            lr_before=opt.param_groups[0]['lr']
            for step,(x,y,image_ids) in enumerate(loader):
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
                use_mix=np.random.random()<.3;lam=float(np.random.beta(.2,.2)) if use_mix else 1.
                order=torch.randperm(len(y),device='cuda') if use_mix else torch.arange(len(y),device='cuda')
                b=y[order];x=lam*x+(1-lam)*x[order] if use_mix else x
                denom=float((lam*w[y]+(1-lam)*w[b]).sum());opt.zero_grad(set_to_none=True)
                active_micro=min(16,c['microbatch']) if precision=='fp32' else c['microbatch']
                for start,end in micro_ranges(len(y),active_micro):
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):logits=model(x[start:end])
                    if not torch.isfinite(logits).all():raise NumericalFailure('Nonfinite training logits')
                    num=mixed_focal_sum(logits,y[start:end],b[start:end],lam,w,2.2)
                    if not torch.isfinite(num):raise NumericalFailure('Nonfinite training focal loss')
                    (num/denom).backward();numerator+=float(num.detach())
                    guessed=logits.argmax(1);correct+=float((lam*(guessed==y[start:end]).float()+(1-lam)*(guessed==b[start:end]).float()).sum())
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):raise NumericalFailure('Nonfinite unscaled BF16/FP32 gradients')
                try:torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
                except RuntimeError as exc:raise NumericalFailure('Nonfinite gradient norm') from exc
                opt.step();updates+=1;denominator+=denom;used+=len(y);mixed_count+=int(use_mix)
                if step%25==0:
                    write_json(folder/'progress.json',dict(status='training',epoch=epoch,step=step+1,steps=len(loader),stage=stage,precision=precision))
                    log.info('epoch %d/50 step %d/%d stage=%s focal_loss=%.5f',epoch,step+1,len(loader),stage,numerator/denominator)
            metrics,preds=evaluate(model,val_loader,w,c);scheduler.step()
            row=dict(epoch=epoch,train_loss=numerator/denominator,train_accuracy=correct/used,train_accuracy_definition='Mixup lambda-weighted target agreement',train_images_used=used,
                val_loss=metrics['loss'],val_accuracy=metrics['accuracy'],val_macro_precision=metrics['macro_precision'],val_macro_recall=metrics['macro_recall'],val_macro_f1=metrics['macro_f1'],
                stage=stage,training_precision=precision,validation_precision='fp32',mixup_batches=mixed_count,backbone_lr=lr_before,head_lr=lr_before,lr_after=scheduler.get_last_lr()[0],
                epoch_seconds=time.perf_counter()-epoch_tick,peak_allocated_vram_mb=torch.cuda.max_memory_allocated()/2**20)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if best is None or metrics['accuracy']>best['metrics']['accuracy']:best=dict(epoch=epoch,metrics=metrics,predictions=preds,model=state)
            if f1best is None or metrics['macro_f1']>f1best['metrics']['macro_f1']:f1best=dict(epoch=epoch,metrics=metrics,predictions=preds,model=state)
            history.append(row);stopping=stopping_status(history);stale=stopping['stale']
            row.update(meaningful_accuracy_reference=stopping['accuracy_reference'],meaningful_macro_f1_reference=stopping['macro_f1_reference'],meaningful_stale=stale,meaningful_accuracy_stale=stopping['accuracy_stale'],meaningful_macro_f1_stale=stopping['macro_f1_stale'],stopping_policy='meaningful_v2')
            atomic_checkpoint(ck/'latest.pt',payload())
            for name,chosen in [('best.pt',best),('best_macro_f1.pt',f1best)]:
                if chosen['epoch']==epoch:atomic_checkpoint(ck/name,dict(config=c,spec=spec,model=chosen['model'],best_epoch=epoch,metrics=chosen['metrics'],source_hashes=hashes()))
            write_csv(folder/'history.csv',history);upsert(registry_row(spec,c,history,'running'))
            write_json(folder/'early_stopping_state.json',stopping)
            log.info('EPOCH %d accuracy=%.6f macroF1=%.6f best=%d stale=%d',epoch,metrics['accuracy'],metrics['macro_f1'],best['epoch'],stale)
            del state
        for name,suffix,chosen in [('best.pt','',best),('best_macro_f1.pt','_macro_f1',f1best)]:
            atomic_checkpoint(ck/name,dict(config=c,spec=spec,model=chosen['model'],best_epoch=chosen['epoch'],metrics=chosen['metrics'],source_hashes=hashes()))
            write_json(folder/f'validation_metrics{suffix}.json',chosen['metrics']);write_csv(folder/f'validation_predictions{suffix}.csv',chosen['predictions'])
            metric_figures(chosen['metrics'],folder/('figures'+suffix),spec['model']+' | enhanced lesion-disjoint validation')
        # Saved last epoch already has its FP32 predictions; no second inference needed.
        if not history:raise RuntimeError('No completed epoch')
        if 'metrics' not in locals():metrics,preds=evaluate(model,val_loader,w,c)
        write_json(folder/'validation_metrics_latest.json',metrics);write_csv(folder/'validation_predictions_latest.csv',preds)
        training_figures(pd.DataFrame(history),folder/'figures',spec['model']+' | enhanced development',selection_metric='accuracy')
        write_csv(folder/'lr_history.csv',[{k:r[k] for k in ['epoch','backbone_lr','lr_after','stage']} for r in history])
        summary=dict(best_epoch=best['epoch'],best_macro_f1_epoch=f1best['epoch'],stopping_epoch=len(history),runtime_seconds=runtime+time.perf_counter()-tick,
            precision=precision,fp32_fallback=bool((folder/'precision_amendment.json').exists()),parameters=sum(p.numel() for p in model.parameters()),optimizer_updates=updates,stop_reason=stopping_status(history)['reason'],meaningful_stopping=stopping_status(history))
        write_json(folder/'early_stopping_state.json',stopping_status(history))
        write_json(folder/'training_summary.json',summary)
        record=registry_row(spec,c,history,'completed');record.update(best_epoch=best['epoch'],checkpoint=relative(ck/'best.pt'),checkpoint_available_local=True,checkpoint_sha256=sha256(ck/'best.pt'),
            metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),runtime_seconds=summary['runtime_seconds'],val_loss=best['metrics']['loss'],**{k:best['metrics'][k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']})
        upsert(record);write_json(folder/'record.json',dict(record,config_sha256=sha256(CONFIG)));log.info('COMPLETED %s',rid)
    except BaseException as exc:
        stamp='failure_'+str(time.time_ns());details=dict(error=repr(exc),numerical=isinstance(exc,NumericalFailure),precision=precision,committed_epoch=len(history),
            model_nonfinite=tensor_nonfinite_names(model.state_dict()),optimizer_nonfinite=tensor_nonfinite_names(opt.state_dict()))
        write_json(folder/(stamp+'.json'),details)
        atomic_checkpoint(folder/(stamp+'.pt'),dict(diagnostic_only=True,**payload(),batch_images=x.detach().cpu() if 'x' in locals() else None,batch_targets=y.detach().cpu() if 'y' in locals() else None))
        write_json(folder/'failure_status.json',details);log.exception('STOP: preserved diagnostics/latest valid checkpoint');raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--fp32',action='store_true');args=ap.parse_args()
    spec=next(s for s in config()['models'] if s['id']==args.id)
    with run_lock(args.id):run(spec,args.resume,args.fp32)

if __name__=='__main__':main()
