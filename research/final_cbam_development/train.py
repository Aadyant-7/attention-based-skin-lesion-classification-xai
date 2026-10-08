"""S79 explicit launch/resume runner. --check never initializes CUDA."""
import argparse
import gc
import json
import logging
import os
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from research.common import ROOT, CLASSES, relative, write_json
from research.strict_train import atomic_checkpoint, run_lock, runtime_versions
from research.short_screening.lesion_bag_screen import report
from .protocol import verified_development, weights_from_training, make_model, make_optimizer, make_scheduler, stage, stopping, check_resume
from .runtime import signature, require_launch_freeze, deterministic_cuda, capture_rng, load_rng, loader, model_cpu, finite, amp_update, NUMERICS
from .artifacts import COLS, repair, closeout


def validation(model,dl,weights):
    model.eval();ys=[];ids=[];arrays=[];numerator=denominator=0.
    with torch.inference_mode():
        for x,y,image_ids in dl:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite validation input')
            z=model(x).float();p=z.softmax(1)
            num=torch.nn.functional.cross_entropy(z,y,weight=weights,reduction='sum')
            if not torch.isfinite(z).all() or not torch.isfinite(p).all() or not torch.isfinite(num):
                raise FloatingPointError('Nonfinite FP32 validation logits/probabilities/loss')
            numerator+=float(num);denominator+=float(weights[y].sum());ys.extend(y.cpu().tolist());ids.extend(image_ids);arrays.append(p.cpu().numpy())
    p=np.concatenate(arrays);m=report(np.array(ys),p);m['loss']=numerator/denominator
    return m,p,ids


def checkpoint(c,sig,model,opt,scheduler,scaler,epoch,history,best,f1,last,updates,runtime):
    return dict(config=c,signature=sig,epoch=epoch,model=model_cpu(model),optimizer=opt.state_dict(),
        scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=capture_rng(),history=history,
        meaningful_stopping=stopping(history,c),best_accuracy=best,best_macro_f1=f1,last_validation=last,
        optimizer_updates=updates,runtime_seconds=runtime,
        loader_recovery='epoch seed=seed+epoch; complete epoch replay after partial interruption; no mid-epoch resume')


def selected(epoch,metrics,predictions,model):
    return dict(epoch=epoch,metrics=metrics,predictions=predictions,model=model_cpu(model))


def run(resume=False):
    c,sig=require_launch_freeze();train,val=verified_development(c);weights,_=weights_from_training(train)
    folder=ROOT/c['results'];ck=ROOT/c['checkpoints'];latest=ck/'latest.pt'
    if resume:
        if not latest.exists():raise FileNotFoundError('No committed checkpoint to resume')
        saved=torch.load(latest,map_location='cpu',weights_only=False);check_resume(saved,c,sig)
        if 'cuda' not in saved['rng']:raise ValueError('Missing CUDA RNG recovery state')
        if (folder/'progress.json').exists() and json.loads((folder/'progress.json').read_text()).get('status')=='completed':
            print('Completed S79 preserved; no GPU initialization or training');return
    else:
        if folder.exists() or ck.exists():raise FileExistsError('Run exists; use explicit --resume')
        saved=None;folder.mkdir(parents=True);ck.mkdir(parents=True)
        write_json(folder/'config.json',c);write_json(folder/'launch_manifest.json',sig)
        write_json(folder/'environment.json',dict(runtime=runtime_versions(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    logger=logging.getLogger('s79');logger.setLevel(logging.INFO)
    handlers=[logging.FileHandler(folder/'train.log'),logging.StreamHandler()]
    for h in handlers:h.setFormatter(logging.Formatter('%(asctime)s %(message)s'));logger.addHandler(h)
    tick=time.perf_counter();base_runtime=(saved or {}).get('runtime_seconds',0)
    epoch=(saved or {}).get('epoch',0);history=(saved or {}).get('history',[]);updates=(saved or {}).get('optimizer_updates',0)
    best=(saved or {}).get('best_accuracy');f1=(saved or {}).get('best_macro_f1');last=(saved or {}).get('last_validation')
    try:
        logger.info('START S79 resume=%s committed_epoch=%d train=%d val=%d FP32 validation; no test loader',resume,epoch,len(train),len(val))
        deterministic_cuda(c['seed']);model=make_model(c,pretrained=not resume).cuda();opt=make_optimizer(model,c)
        scheduler=make_scheduler(opt,c);scaler=torch.amp.GradScaler('cuda',init_scale=NUMERICS['gradscaler_initial_scale']);weights=weights.cuda()
        if saved:
            model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
            load_rng(saved['rng']);repair(saved,folder,ck,c);del saved;gc.collect()
        else:
            initial=checkpoint(c,sig,model,opt,scheduler,scaler,0,[],None,None,None,0,0.)
            atomic_checkpoint(latest,initial);repair(initial,folder,ck,c);del initial
        write_json(folder/'process.json',dict(pid=os.getpid(),started=True,results=c['results'],checkpoints=c['checkpoints'],resume=resume))
        logger.info('LOGGING/CHECKPOINTING ACTIVE latest=%s',relative(latest))
        for next_epoch in range(epoch+1,c['max_epochs']+1):
            if stopping(history,c)['stop']:break
            epoch=next_epoch;start=time.perf_counter();torch.cuda.reset_peak_memory_stats();phase=stage(model,epoch,c)
            dl=loader(train,c,epoch,True);assert len(dl)%c['gradient_accumulation']==0
            iterator=iter(dl);num=den=0.;correct=used=0;lr_before=[g['lr'] for g in opt.param_groups]
            for i in range(len(dl)//c['gradient_accumulation']):
                batches=[]
                for _ in range(c['gradient_accumulation']):
                    x,y,_ids=next(iterator);batches.append((x.cuda(non_blocking=True),y.cuda(non_blocking=True)))
                stats=amp_update(model,opt,scaler,batches,weights,c);updates+=1
                num+=stats['numerator'];den+=stats['denominator'];correct+=stats['correct'];used+=stats['used']
                if i==0 or (i+1)%25==0:logger.info('epoch %d/%d update %d/%d stage=%s',epoch,c['max_epochs'],i+1,len(dl)//2,phase)
            del dl,iterator,batches;gc.collect()
            metrics,p,ids=validation(model,loader(val,c,epoch,False),weights)
            if ids!=val.image_id.tolist():raise ValueError('Validation IDs/order mismatch')
            frame=val[['image_id','lesion_id']].copy();frame['true_class']=val.diagnosis;frame['predicted_class']=[CLASSES[k] for k in p.argmax(1)];frame[COLS]=p
            predictions=frame.to_dict('records');last=dict(epoch=epoch,metrics=metrics,predictions=predictions)
            if epoch>c['warmup_epochs']:scheduler.step(metrics['accuracy'])
            lr_after=[g['lr'] for g in opt.param_groups]
            row=dict(epoch=epoch,epoch_seed=c['seed']+epoch,stage=phase,train_loss=num/den,train_accuracy=correct/used,
                train_images_used=used,val_loss=metrics['loss'],**{'val_'+k:metrics[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']},
                backbone_lr=lr_before[0],head_lr=lr_before[1],backbone_lr_after=lr_after[0],head_lr_after=lr_after[1],
                scheduler_lr_reduced=any(a<b for a,b in zip(lr_after,lr_before)),optimizer_updates=updates,
                scaler_scale=scaler.get_scale(),training_precision='amp_fp16',validation_precision='fp32',
                epoch_seconds=time.perf_counter()-start,peak_allocated_vram_mb=torch.cuda.max_memory_allocated()/1024**2)
            history.append(row);policy=stopping(history,c);row.update(meaningful_stale=policy['stale'],last_meaningful_epoch=policy['last_meaningful_epoch'],
                meaningful_accuracy_reference=policy['references']['accuracy'],meaningful_macro_f1_reference=policy['references']['macro_f1'],stopping_reason=policy['reason'])
            if best is None or metrics['accuracy']>best['metrics']['accuracy']:best=selected(epoch,metrics,predictions,model)
            if f1 is None or metrics['macro_f1']>f1['metrics']['macro_f1']:f1=selected(epoch,metrics,predictions,model)
            payload=checkpoint(c,sig,model,opt,scheduler,scaler,epoch,history,best,f1,last,updates,base_runtime+time.perf_counter()-tick)
            atomic_checkpoint(latest,payload) # Commit before any CSV/registry/figure mutation.
            repair(payload,folder,ck,c);del payload
            logger.info('EPOCH %d accuracy=%.6f macroF1=%.6f best_accuracy_epoch=%d best_F1_epoch=%d meaningful_stale=%d stop=%s',
                        epoch,metrics['accuracy'],metrics['macro_f1'],best['epoch'],f1['epoch'],policy['stale'],policy['reason'])
        # Closeout reads the committed state; no further inference or epoch selection.
        committed=torch.load(latest,map_location='cpu',weights_only=False);check_resume(committed,c,sig)
        closeout(committed,folder,ck,c,val)
        logger.info('COMPLETED S79 and one fixed CPU ensemble; committed_epochs=%d; no additional experiment',committed['epoch'])
    except BaseException as exc:
        stamp='failure_'+str(time.time_ns())
        info=dict(error=repr(exc),attempted_epoch=epoch,latest_valid_checkpoint=relative(latest) if latest.exists() else None,
            latest_committed_epoch=None,automatic_retry=False)
        if latest.exists():info['latest_committed_epoch']=torch.load(latest,map_location='cpu',weights_only=False)['epoch']
        write_json(folder/(stamp+'.json'),info)
        if isinstance(exc,(FloatingPointError,RuntimeError)) and 'model' in locals():
            atomic_checkpoint(ck/(stamp+'_diagnostic.pt'),dict(diagnostic_only=True,config=c,signature=sig,
                attempted_epoch=epoch,model=model_cpu(model),optimizer=opt.state_dict(),scaler=scaler.state_dict()))
        write_json(folder/'progress.json',dict(status='failed_preserved',**info));logger.exception('STOPPED; latest valid epoch preserved');raise
    finally:
        for h in handlers:logger.removeHandler(h);h.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--start',action='store_true');mode.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.check:
        c,s=require_launch_freeze();train,val=verified_development(c)
        from .artifacts import fp32_parts
        fp32_parts(s,val)
        print(json.dumps(dict(status='ready_pending_training_approval',train=len(train),validation=len(val),test_loader=False,
            cuda_initialized=torch.cuda.is_initialized(),results=c['results'],checkpoints=c['checkpoints'],max_epochs=c['max_epochs'])))
    else:
        with run_lock('s79_convnext_tiny_cbam_exploratory_seed42'):run(a.resume)


if __name__=='__main__':main()
