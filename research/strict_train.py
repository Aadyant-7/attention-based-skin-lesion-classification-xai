"""Controlled train/validation runner. No test loader; CLI check never uses CUDA.

Launching training is a separate human-approved action. --check is read-only.
Resume is from the last atomic epoch boundary, not from a partially trained epoch.
"""
import argparse
import copy
import json
import logging
import os
import random
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from .common import ROOT, CLASSES, SPLITS, sha256, write_csv, write_json
from .strict_experiment import Experiment
from .models import CHANNELS, ResearchClassifier

RECIPE = ROOT/'research/phase2/recipe_v1.json'
RECIPES={'final_strict_v1':ROOT/'research/configs/final_strict_recipe_v1.json'}


def validate_config(config):
    if config.get('recipe_version') not in RECIPES:
        raise ValueError('Unregistered recipe version')
    recipe=json.loads(RECIPES[config['recipe_version']].read_text(encoding='utf-8'))
    differences=[k for k,v in recipe.items() if k!='status' and config.get(k)!=v]
    if differences:
        raise ValueError(f'Unsupported recipe amendment: {differences}; register/review a new runner recipe first')
    if config.get('model') not in CHANNELS or not re.fullmatch(r's\d\d_[a-z0-9_]{3,95}',config.get('experiment_id','')):
        raise ValueError('Unsupported model or structured ID')
    enum=models.get_model_weights(config['model'])
    if config['weights'].split('.')[0]!=enum.__name__ or config['weights'].split('.')[-1]=='DEFAULT':
        raise ValueError('Explicit matching weights required')
    enum[config['weights'].split('.')[-1]]
    if not config.get('question'):
        raise ValueError('A research question is required')


from .strict_protocol import development_data, partition_metadata


class DevelopmentImages(Dataset):
    def __init__(self, frame, config, training=False):
        if not set(frame.split).issubset({'train'} if training else {'val'}):
            raise ValueError('Test or wrong partition cannot enter a development loader')
        self.frame=frame
        self.transform=transforms.Compose([
            transforms.Resize((config['image_size'],)*2,interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
            *([transforms.RandomHorizontalFlip(.5)] if training else []),
            transforms.ToTensor(),transforms.Normalize(config['normalization_mean'],config['normalization_std'])])

    def __len__(self): return len(self.frame)

    def __getitem__(self,index):
        row=self.frame.iloc[index]
        with Image.open(row.path) as im: image=self.transform(im.convert('RGB'))
        return image,int(row.label),str(row.image_id)


def seed_worker(_):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed)


def optimizer_groups(model,config):
    fresh={id(p) for layer in (model.attention,model.head[-1]) for p in layer.parameters()}
    backbone=[p for p in model.parameters() if id(p) not in fresh]
    head=[p for p in model.parameters() if id(p) in fresh]
    return [{'params':backbone,'lr':config['learning_rate']['backbone']},
            {'params':head,'lr':config['learning_rate']['head']}]


def weighted_numerator(logits, targets, weights):
    return F.cross_entropy(logits.float(),targets,weight=weights,reduction='sum')


class NonfiniteValidation(FloatingPointError):
    def __init__(self, details):
        self.details=details
        super().__init__('Nonfinite validation batch: '+json.dumps(details))


def checked_validation_batch(model, images, targets, weights, fp32=False):
    """Never let invalid outputs reach argmax/metrics; forward precision is explicit."""
    if not torch.isfinite(images).all() or not torch.isfinite(weights).all() or (weights<=0).any():
        raise NonfiniteValidation(dict(reason='invalid_inputs_or_weights'))
    if (targets<0).any() or (targets>=len(CLASSES)).any():
        raise NonfiniteValidation(dict(reason='invalid_targets'))
    with torch.autocast(images.device.type,dtype=torch.float16,enabled=not fp32):
        logits=model(images)
    num=weighted_numerator(logits,targets,weights)
    probabilities=logits.float().softmax(1)
    if not (torch.isfinite(logits).all() and torch.isfinite(num) and torch.isfinite(probabilities).all()):
        raise NonfiniteValidation(dict(reason='nonfinite_forward_or_loss',precision='fp32' if fp32 else 'amp_fp16',
            logits_nan=int(torch.isnan(logits).sum()),logits_inf=int(torch.isinf(logits).sum()),
            bad_logit_rows=(~torch.isfinite(logits).all(1)).nonzero().flatten().cpu().tolist(),
            loss_finite=bool(torch.isfinite(num)),probabilities_nonfinite=int((~torch.isfinite(probabilities)).sum())))
    return num,probabilities


def tensor_nonfinite_names(obj,prefix=''):
    if torch.is_tensor(obj):
        return [prefix] if obj.is_floating_point() and not torch.isfinite(obj).all() else []
    if isinstance(obj,dict):
        return sum((tensor_nonfinite_names(v,prefix+'.'+str(k)) for k,v in obj.items()),[])
    if isinstance(obj,(tuple,list)):
        return sum((tensor_nonfinite_names(v,prefix+'.'+str(i)) for i,v in enumerate(obj)),[])
    return []


def evaluate_validation(model,loader,weights,config,path,epoch,allow_fp32_recovery=False,training_state=None):
    """Unchanged AMP normally; approved S10 recovery may repeat a whole pass in FP32."""
    device=next(model.parameters()).device
    model.eval()
    def collect(fp32):
        all_y=[];all_p=[];ids=[];vn=vd=0.
        with torch.inference_mode():
            for batch, (images,y,image_ids) in enumerate(loader):
                images=images.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                try:num,p=checked_validation_batch(model,images,y,weights,fp32)
                except NonfiniteValidation as exc:
                    exc.batch_data=(images,y,list(image_ids),batch)
                    raise
                vn+=float(num);vd+=float(weights[y].sum())
                all_y.extend(y.cpu().tolist());all_p.extend(p.cpu().tolist());ids.extend(image_ids)
        if not np.isfinite(vn/vd):raise FloatingPointError('Nonfinite validation accumulation')
        return all_y,all_p,ids,vn/vd,'fp32' if fp32 else 'amp_fp16'
    try:return collect(True)
    except NonfiniteValidation as exc:
        images,y,image_ids,batch=exc.batch_data
        stamp=f'validation_failure_epoch{epoch}_{time.time_ns()}'
        details=dict(exc.details,epoch=epoch,batch=batch,image_ids=image_ids,
            model_nonfinite_tensors=tensor_nonfinite_names(model.state_dict()),
            optimizer_nonfinite_tensors=tensor_nonfinite_names((training_state or {}).get('optimizer',{})),
            fp32_recovery_authorized=allow_fp32_recovery)
        # Diagnostic snapshot only; latest.pt remains the valid resume boundary.
        atomic_checkpoint(path/(stamp+'.pt'),dict(config=config,epoch=epoch,model=model.state_dict(),
            code_hashes=code_hashes(),diagnostic_only=True,**(training_state or {})))
        handles=[];first_bad=[]
        def hook(name):
            def inspect(_module,_input,output):
                if torch.is_tensor(output) and not first_bad and not torch.isfinite(output).all():
                    first_bad.append(dict(module=name,dtype=str(output.dtype),nan=int(torch.isnan(output).sum()),inf=int(torch.isinf(output).sum())))
            return inspect
        for name,module in model.named_modules():
            if not list(module.children()):handles.append(module.register_forward_hook(hook(name)))
        try:
            with torch.inference_mode():
                try:checked_validation_batch(model,images,y,weights)
                except NonfiniteValidation:pass
        finally:
            for handle in handles:handle.remove()
        details['amp_replay_first_nonfinite_module']=first_bad
        can_retry=(allow_fp32_recovery and exc.details['reason']=='nonfinite_forward_or_loss'
                   and not details['model_nonfinite_tensors'] and not details['optimizer_nonfinite_tensors'])
        details['fp32_probe_finite']=False
        if can_retry:
            try:
                with torch.inference_mode():checked_validation_batch(model,images,y,weights,fp32=True)
                details['fp32_probe_finite']=True
            except NonfiniteValidation as fp32_exc:details['fp32_probe_failure']=fp32_exc.details
        details['decision']='repeat_entire_validation_fp32' if details['fp32_probe_finite'] else 'stop_preserve_valid_checkpoint'
        write_json(path/(stamp+'.json'),details)
        if not details['fp32_probe_finite']:raise
        try:return collect(True)
        except NonfiniteValidation as fp32_exc:
            images,y,image_ids,batch=fp32_exc.batch_data
            write_json(path/(stamp+'_fp32_failed.json'),dict(fp32_exc.details,epoch=epoch,batch=batch,image_ids=image_ids))
            raise


def metric_report(targets, probabilities, loss):
    from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
    predicted=np.asarray(probabilities).argmax(axis=1)
    p,r,f,s=precision_recall_fscore_support(targets,predicted,labels=range(7),zero_division=0)
    return dict(loss=float(loss),accuracy=float(np.mean(predicted==targets)),
        macro_precision=float(p.mean()),macro_recall=float(r.mean()),macro_f1=float(f.mean()),
        class_order=list(CLASSES),confusion_matrix=confusion_matrix(targets,predicted,labels=range(7)).tolist(),
        per_class={c:dict(precision=float(p[i]),recall=float(r[i]),f1=float(f[i]),support=int(s[i])) for i,c in enumerate(CLASSES)})


def atomic_checkpoint(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        Path(name).unlink(missing_ok=True)


def code_hashes():
    return {f:sha256(ROOT/f) for f in ('research/strict_train.py','research/strict_protocol.py','research/run_final_strict_training.py','research/final_strict_results.py','research/configs/final_strict_recipe_v1.json','research/models.py','research/strict_experiment.py',
        'research/common.py','research/plots.py','research/registry.py','src/cbam.py')}


def runtime_versions():
    return dict(python=sys.version,**{p:version(p) for p in ('torch','torchvision','numpy','pandas','Pillow','scikit-learn')})


@contextmanager
def run_lock(experiment_id):
    """OS lock prevents two resumes; OS releases it even after a killed process."""
    path=ROOT/'.cache/research_run_locks'/f'{experiment_id}.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0,2)
        if f.tell()==0: f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def rng_state(cuda=False):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if cuda else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def validate_resume_fix(config,checkpoint,fix):
    """Only this reviewed S10 source migration may bypass the original source match."""
    rid='s10_efficientnet_v2_s_none_exploratory_seed42'
    if (fix.get('kind')!='s10_validation_numerics_v1' or config['experiment_id']!=rid
            or fix.get('experiment_id')!=rid or fix.get('checkpoint_epoch')!=len(checkpoint['history'])
            or len(checkpoint['history'])!=13 or fix.get('from_code_hashes')!=checkpoint['code_hashes']
            or fix.get('to_code_hashes')!=code_hashes()
            or {k for k in checkpoint['code_hashes'] if checkpoint['code_hashes'][k]!=code_hashes().get(k)}!={'research/train.py'}
            or fix.get('config_sha256')!=sha256(ROOT/'research/configs/phase3'/f'{rid}.json')
            or fix.get('checkpoint_sha256')!=sha256(ROOT/'checkpoints/structured'/rid/'latest.pt')):
        raise ValueError('Unapproved or mismatched S10 resume-fix manifest')


def restore_experiment(config, checkpoint, resume_fix=None):
    """Checkpoint is the authoritative committed epoch, repairing partial CSV writes."""
    from .registry import upsert
    path=ROOT/'results/structured_experiments'/config['experiment_id']
    saved=json.loads((path/'config.json').read_text(encoding='utf-8'))
    if (saved!=config or checkpoint['config']!=config or checkpoint['runtime_versions']!=runtime_versions()):
        raise ValueError('Resume config or runner changed; do not silently continue another recipe')
    if resume_fix is not None:validate_resume_fix(config,checkpoint,resume_fix)
    elif checkpoint['code_hashes']!=code_hashes():
        raise ValueError('Resume config or runner changed; an explicit reviewed resume-fix manifest is required')
    obj=Experiment.__new__(Experiment);obj.config=saved;obj.path=path
    obj.history=checkpoint['history'];obj.row=checkpoint['registry_row']
    obj.validation_support=partition_metadata()[0].query("split=='val'").diagnosis.value_counts().to_dict()
    write_csv(path/'history.csv',obj.history)
    obj.row.update(status='running',epochs=len(obj.history),history_path=(path/'history.csv').relative_to(ROOT).as_posix());upsert(obj.row)
    return obj


def completed_run(config):
    path=ROOT/'results/structured_experiments'/config['experiment_id']
    record=path/'record.json'
    if record.exists() and json.loads(record.read_text(encoding='utf-8')).get('status')=='completed':
        if json.loads((path/'config.json').read_text(encoding='utf-8'))!=config:
            raise ValueError('Completed ID belongs to another config')
        return True
    return False


def save_screening_snapshot(experiment,best,secondary_best,epoch):
    """Publish pilot artifacts without completing or changing the experiment recipe."""
    from .plots import metric_figures, training_figures
    folder=experiment.path/'screening'/f'epoch_{epoch:02d}'
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'metrics.json',best['metrics'])
    write_csv(folder/'predictions.csv',best['predictions'])
    metric_figures(best['metrics'],folder/'figures','Pilot accuracy winner | exploratory validation')
    training_figures(pd.DataFrame(experiment.history),folder/'figures','Pilot training through screening boundary',selection_metric=experiment.config['selection_metric'])
    if secondary_best:
        write_json(folder/'metrics_macro_f1.json',secondary_best['metrics'])
        write_csv(folder/'predictions_macro_f1.csv',secondary_best['predictions'])
    write_json(folder/'screening_status.json',dict(status='paused_screening',committed_epoch=epoch,
        best_epoch=best['epoch'],secondary_best_epoch=secondary_best['epoch'] if secondary_best else None,
        continuation_requires_approval=True,test_loader=False))
    experiment.row.update(status='paused_screening',epochs=epoch,decision='await_pilot_review')
    from .registry import upsert
    upsert(experiment.row);write_json(experiment.path/'record.json',experiment.row)


def run(config,resume=False,resume_fix=None,stop_after_epoch=None):
    if stop_after_epoch is not None and not 1 <= stop_after_epoch < config['max_epochs']:
        raise ValueError('Screening boundary must precede the full epoch cap')
    # Safe completed-run exit occurs before CUDA queries or loading images/weights.
    if completed_run(config):
        print('Already completed; preserved without training.',flush=True);return
    os.environ['CUBLAS_WORKSPACE_CONFIG']=config['cuda_workspace_config']
    if not torch.cuda.is_available(): raise RuntimeError('Approved runner requires CUDA; CPU training is not a fallback')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    train,val,weights=development_data(config)
    ckptdir=ROOT/'checkpoints/structured'/config['experiment_id'];latest=ckptdir/'latest.pt'
    checkpoint=None
    if resume:
        # Our own trusted local full-state checkpoint, never an arbitrary downloaded file.
        checkpoint=torch.load(latest,map_location='cpu',weights_only=False)
        experiment=restore_experiment(config,checkpoint,resume_fix)
    else:
        if ckptdir.exists(): raise FileExistsError('Existing checkpoint directory; inspect/resume explicitly')
        experiment=Experiment(config)
    logger=logging.getLogger(config['experiment_id']);logger.setLevel(logging.INFO);logger.propagate=False
    handlers=[logging.FileHandler(experiment.path/'train.log',encoding='utf-8'),logging.StreamHandler()]
    for h in handlers:
        h.setFormatter(logging.Formatter('%(asctime)s %(message)s'));logger.addHandler(h)
    def progress(**values):
        write_json(experiment.path/'progress.json',dict(experiment_id=config['experiment_id'],**values))
    start=time.perf_counter();runtime=checkpoint['runtime_seconds'] if checkpoint else 0
    amendment=resume_fix or (checkpoint or {}).get('resume_fix')
    if amendment:
        if (amendment['experiment_id']!=config['experiment_id'] or amendment['to_code_hashes']!=code_hashes()
                or amendment['kind']!='s10_validation_numerics_v1'):
            raise ValueError('Stored validation recovery amendment no longer matches this runner')
        write_json(experiment.path/'resume_amendment.json',amendment)
    try:
        logger.info('START %s resume=%s train=%d val=%d protocol=%s selection=%s; test loader absent',config['experiment_id'],resume,len(train),len(val),config['protocol'],config['selection_metric'])
        progress(status='initializing',epoch=len(experiment.history),max_epochs=config['max_epochs'])
        # Resume loads trained weights; no ImageNet download on that path.
        model=ResearchClassifier(config['model'],weights=None if resume else config['weights'],
                                  attention=config['attention'],dropout=config['head_dropout']).cuda()
        optimizer=torch.optim.AdamW(optimizer_groups(model,config),betas=tuple(config['optimizer_betas']),
                                   eps=config['optimizer_eps'],weight_decay=config['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,
                                   threshold=0,threshold_mode='abs',cooldown=0,min_lr=1e-7)
        scaler=torch.amp.GradScaler('cuda')
        weights=weights.cuda();best=None;secondary_best=None;stale=0;updates=0
        selection=config['selection_metric']
        secondary=config.get('secondary_selection_metric')
        if checkpoint:
            model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler']);scaler.load_state_dict(checkpoint['scaler'])
            best=checkpoint['best'];stale=checkpoint['stale'];updates=checkpoint['optimizer_updates']
            secondary_best=checkpoint.get('secondary_best')
            restore_rng(checkpoint['rng'])
        else:
            enum=models.get_model_weights(config['model'])[config['weights'].split('.')[-1]]
            cached=Path(torch.hub.get_dir())/'checkpoints'/enum.url.rsplit('/',1)[-1]
            write_json(experiment.path/'pretraining.json',dict(weights=config['weights'],url=enum.url,
                       sha256=sha256(cached),parameters=sum(p.numel() for p in model.parameters()),
                       metadata_scope='train/validation metadata only; test labels excluded',fresh_imagenet_initialization=True,exploratory_checkpoint_reused=False,
                       code_hashes=code_hashes(),torch=torch.__version__,device=torch.cuda.get_device_name(0),
                       microbatch=config['batch_size'],effective_batch=config['effective_batch_size'],
                       native_attention='SE retained for EfficientNet/MobileNet; none means no added CBAM'))
        trainset=DevelopmentImages(train,config,True);valset=DevelopmentImages(val,config)
        val_loader=DataLoader(valset,batch_size=config['validation_batch_size'],shuffle=False,num_workers=config['workers'],
                              pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(config['seed']))
        if stop_after_epoch is not None and len(experiment.history)>=stop_after_epoch:
            raise ValueError('Screening boundary already reached; choose a later approved boundary')
        for epoch in range(len(experiment.history)+1,config['max_epochs']+1):
            if stale>=config['patience'] and epoch-1>=config['minimum_epochs']: break
            tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();updates_before=updates
            epoch_seed=config['seed']+epoch
            loader=DataLoader(trainset,batch_size=config['batch_size'],shuffle=True,drop_last=True,
                              num_workers=config['workers'],pin_memory=True,worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(epoch_seed))
            model.train();numerator=denominator=0.;correct=used=0;iterator=iter(loader)
            steps=len(loader)//config['gradient_accumulation']
            if len(loader)%config['gradient_accumulation']: raise ValueError('Recipe requires whole effective batches')
            for step in range(steps):
                batches=[next(iterator) for _ in range(config['gradient_accumulation'])]
                total_weight=sum(float(weights[y.cuda()].sum().item()) for _,y,_ in batches)
                optimizer.zero_grad(set_to_none=True)
                for images,y,_ in batches:
                    images=images.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
                    with torch.autocast('cuda',dtype=torch.float16): logits=model(images)
                    num=weighted_numerator(logits,y,weights)
                    if not torch.isfinite(num): raise FloatingPointError('Nonfinite training loss')
                    scaler.scale(num/total_weight).backward()
                    numerator+=float(num.detach());denominator+=float(weights[y].sum())
                    correct+=int((logits.argmax(1)==y).sum());used+=len(y)
                scaler.unscale_(optimizer)
                gradients_finite=bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all())
                if gradients_finite:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)
                else:
                    logger.warning('AMP scaled-gradient overflow: GradScaler will reject optimizer update and reduce scale; no input samples removed; epoch=%d step=%d scale=%g',epoch,step+1,scaler.get_scale())
                scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
                updates+=int(scaler.get_scale()>=scale_before)
                if step%20==0 or step+1==steps:
                    progress(status='training',epoch=epoch,step=step+1,steps=steps,max_epochs=config['max_epochs'],
                             selection_metric=selection,best_epoch=best['epoch'] if best else None,
                             best_accuracy=best['metrics']['accuracy'] if best else None,
                             selected_macro_f1=best['metrics']['macro_f1'] if best else None,
                             best_macro_f1=(secondary_best or best)['metrics']['macro_f1'] if (secondary_best or best) else None)
                    logger.info('epoch %d/%d step %d/%d weighted_loss %.5f',epoch,config['max_epochs'],step+1,steps,numerator/denominator)
            model.eval()
            progress(status='validating',epoch=epoch,max_epochs=config['max_epochs'])
            vt=time.perf_counter()
            all_y,all_p,ids,val_loss,validation_precision=evaluate_validation(model,val_loader,weights,config,experiment.path,epoch,
                allow_fp32_recovery=bool(amendment),training_state=dict(optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(True)))
            if validation_precision=='fp32':logger.info('Validation epoch %d used predefined FP32 validation',epoch)
            validation_ms=(time.perf_counter()-vt)*1000/len(val)
            metrics=metric_report(np.asarray(all_y),all_p,val_loss)
            if not np.isfinite(metrics['loss']): raise FloatingPointError('Nonfinite validation loss')
            lr_before=[g['lr'] for g in optimizer.param_groups];scheduler.step(metrics[selection])
            row=dict(epoch=epoch,train_loss=numerator/denominator,train_accuracy=correct/used,train_images_used=used,
                     val_loss=metrics['loss'],val_accuracy=metrics['accuracy'],val_macro_precision=metrics['macro_precision'],
                     val_macro_recall=metrics['macro_recall'],val_macro_f1=metrics['macro_f1'],epoch_seed=epoch_seed,
                     backbone_lr=lr_before[0],head_lr=lr_before[1],backbone_lr_after=optimizer.param_groups[0]['lr'],head_lr_after=optimizer.param_groups[1]['lr'],
                     scheduler_lr_reduced=optimizer.param_groups[0]['lr']<lr_before[0],optimizer_updates=updates,amp_skipped_optimizer_updates=steps-(updates-updates_before),
                     validation_ms_per_image=validation_ms,peak_allocated_vram_mb=torch.cuda.max_memory_allocated()/1024**2,
                     epoch_seconds=time.perf_counter()-tick,validation_precision=validation_precision)
            predictions=[dict(image_id=i,true_class=CLASSES[y],predicted_class=CLASSES[int(np.argmax(p))],
                              **{f'p_{c}':float(p[j]) for j,c in enumerate(CLASSES)}) for i,y,p in zip(ids,all_y,all_p)]
            improved=best is None or metrics[selection]>best['metrics'][selection]
            stale=0 if improved else stale+1
            if improved:
                best=dict(epoch=epoch,metrics=metrics,predictions=predictions,validation_precision=validation_precision,
                          model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            secondary_improved=secondary and (secondary_best is None or metrics[secondary]>secondary_best['metrics'][secondary])
            if secondary_improved:
                secondary_best=dict(epoch=epoch,metrics=metrics,predictions=predictions,validation_precision=validation_precision,
                                    model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            # Commit full state first. If CSV/registry/plots fail, explicit resume repairs
            # them from this snapshot without rerunning an already committed epoch.
            payload=dict(config=config,code_hashes=code_hashes(),runtime_versions=runtime_versions(),model=model.state_dict(),optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(True),best=best,secondary_best=secondary_best,stale=stale,
                         history=experiment.history+[row],registry_row=copy.deepcopy(experiment.row),
                         optimizer_updates=updates,runtime_seconds=runtime+time.perf_counter()-start,resume_fix=amendment)
            atomic_checkpoint(latest,payload)
            if improved: atomic_checkpoint(ckptdir/'best.pt',dict(payload,model=best['model']))
            if secondary_improved:
                atomic_checkpoint(ckptdir/f'best_{secondary}.pt',dict(config=config,model=secondary_best['model'],
                          best_epoch=secondary_best['epoch'],metrics=secondary_best['metrics'],class_order=list(CLASSES),selection_metric=secondary))
            experiment.log_epoch(row)
            logger.info('EPOCH %d val_accuracy=%.6f macro_f1=%.6f best_epoch=%d stale=%d',epoch,metrics['accuracy'],metrics['macro_f1'],best['epoch'],stale)
            if stop_after_epoch is not None and epoch>=stop_after_epoch:
                save_screening_snapshot(experiment,best,secondary_best,epoch)
                progress(status='paused_screening',epochs=epoch,best_epoch=best['epoch'])
                logger.info('SCREENING PAUSED: committed checkpoint; continuation requires approval')
                return
        if best is None: raise RuntimeError('No committed epoch; cannot publish a completed result')
        final_y,final_p,final_ids,final_loss,_=evaluate_validation(model,val_loader,weights,config,experiment.path,len(experiment.history))
        final_metrics=metric_report(np.asarray(final_y),final_p,final_loss)
        final_predictions=[dict(image_id=i,true_class=CLASSES[y],predicted_class=CLASSES[int(np.argmax(p))],
            **{f'p_{c}':float(p[j]) for j,c in enumerate(CLASSES)}) for i,y,p in zip(final_ids,final_y,final_p)]
        write_json(experiment.path/'validation_metrics_latest.json',final_metrics)
        write_csv(experiment.path/'validation_predictions_latest.csv',final_predictions)
        from .plots import metric_figures
        metric_figures(final_metrics,experiment.path/'figures_latest','Final strict validation | latest checkpoint')
        if secondary_best:
            metric_figures(secondary_best['metrics'],experiment.path/'figures_macro_f1','Final strict validation | macro-F1 checkpoint')
        write_csv(experiment.path/'lr_history.csv',[{k:r[k] for k in ('epoch','backbone_lr','head_lr','backbone_lr_after','head_lr_after','scheduler_lr_reduced')} for r in experiment.history])

        # This also repairs a best.pt write interrupted after latest.pt committed.
        final=dict(config=config,code_hashes=code_hashes(),model=best['model'],best_epoch=best['epoch'],
                   metrics=best['metrics'],class_order=list(CLASSES),selection_metric=selection)
        atomic_checkpoint(ckptdir/'best.pt',final)
        write_csv(experiment.path/'validation_predictions.csv',best['predictions'])
        if secondary_best:
            atomic_checkpoint(ckptdir/f'best_{secondary}.pt',dict(config=config,model=secondary_best['model'],
                 best_epoch=secondary_best['epoch'],metrics=secondary_best['metrics'],class_order=list(CLASSES),selection_metric=secondary))
            write_json(experiment.path/f'validation_metrics_{secondary}.json',secondary_best['metrics'])
            write_csv(experiment.path/f'validation_predictions_{secondary}.csv',secondary_best['predictions'])
        write_json(experiment.path/'training_summary.json',dict(parameters=sum(p.numel() for p in model.parameters()),
                   optimizer_updates=updates,best_epoch=best['epoch'],selection_metric=selection,
                   secondary_best_epoch=secondary_best['epoch'] if secondary_best else None,
                   stopping_epoch=len(experiment.history),runtime_seconds=runtime+time.perf_counter()-start,maximum_epochs=config['max_epochs'],minimum_epochs=config['minimum_epochs'],
                   fp32_validation_fallback_used=False,validation_precision='fp32_from_start',gradient_clip_norm=config['gradient_clip_norm'],
                   amp_skipped_optimizer_updates=sum(r['amp_skipped_optimizer_updates'] for r in experiment.history),
                   stop_reason='patience' if stale>=config['patience'] else 'epoch_cap',
                   validation_timing_scope='End-to-end validation batch loop including loading/transfers; not pure GPU latency',
                   best_validation_ms_per_image=experiment.history[best['epoch']-1]['validation_ms_per_image'],
                   validation_recovery_amendment=amendment,
                   selected_validation_precision=best.get('validation_precision','amp_fp16')))
        experiment.complete(best['metrics'],ckptdir/'best.pt',best['epoch'],runtime+time.perf_counter()-start,
                            f"{config['recipe_version']}; {config['protocol']} validation only; selected earliest maximum {selection}; no test loader."
                            + (' Explicit S10 numerical recovery amendment: training AMP unchanged; any recovered validation epochs use full FP32, recorded in history/diagnostics.' if amendment else ''))
        progress(status='completed',epochs=len(experiment.history),best_epoch=best['epoch'],
                 selection_metric=selection,best_accuracy=best['metrics']['accuracy'],selected_macro_f1=best['metrics']['macro_f1'],
                 best_macro_f1=(secondary_best or best)['metrics']['macro_f1'])
        logger.info('COMPLETED best_epoch=%d accuracy=%.6f macro_f1=%.6f',best['epoch'],best['metrics']['accuracy'],best['metrics']['macro_f1'])
    except BaseException as exc:
        stamp=f'training_failure_{time.time_ns()}'
        write_json(experiment.path/(stamp+'.json'),dict(error=repr(exc),committed_epochs=len(experiment.history),
            model_nonfinite_tensors=tensor_nonfinite_names(model.state_dict()) if 'model' in locals() else [],
            optimizer_nonfinite_tensors=tensor_nonfinite_names(optimizer.state_dict()) if 'optimizer' in locals() else []))
        if 'model' in locals():
            atomic_checkpoint(experiment.path/(stamp+'.pt'),dict(diagnostic_only=True,model=model.state_dict(),
                optimizer=optimizer.state_dict() if 'optimizer' in locals() else {},config=config,code_hashes=code_hashes()))
        experiment.fail(repr(exc));progress(status='failed',error=repr(exc),committed_epochs=len(experiment.history))
        logger.exception('FAILED; committed checkpoints preserved');raise
    finally:
        for handler in handlers: logger.removeHandler(handler);handler.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--check',action='store_true',help='Read-only config/data-path check; no model, CUDA, downloads or run outputs')
    parser.add_argument('--resume',action='store_true',help='Resume trusted latest.pt at committed epoch boundary')
    parser.add_argument('--resume-fix',type=Path,help='Explicit reviewed S10 numerical-recovery manifest; requires --resume')
    parser.add_argument('--stop-after-epoch',type=int,help='Pause at a committed pilot epoch; recipe and resume state remain unchanged')
    args=parser.parse_args()
    if args.resume_fix and not args.resume:parser.error('--resume-fix requires --resume')
    config=json.loads(args.config.read_text(encoding='utf-8'));validate_config(config)
    if args.check:
        train,val,w=development_data(config)
        print(json.dumps(dict(status='ready_authorized_final_strict',experiment_id=config['experiment_id'],
                   config_sha256=sha256(args.config),train_images=len(train),validation_images=len(val),
                   class_weights=dict(zip(CLASSES,w.tolist())),test_loader=False,cuda_used=False,
                   checkpoints=f"checkpoints/structured/{config['experiment_id']}",
                   results=f"results/structured_experiments/{config['experiment_id']}"),indent=2))
    else:
        fix=json.loads(args.resume_fix.read_text(encoding='utf-8')) if args.resume_fix else None
        with run_lock(config['experiment_id']): run(config,args.resume,fix,args.stop_after_epoch)


if __name__=='__main__': main()
