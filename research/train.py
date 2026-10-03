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
from .common import ROOT, CLASSES, sha256, write_csv, write_json
from .experiment import Experiment
from .models import CHANNELS, ResearchClassifier

RECIPE = ROOT/'research/phase2/recipe_v1.json'


def validate_config(config):
    recipe=json.loads(RECIPE.read_text(encoding='utf-8'))
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


def development_data(config, root=ROOT):
    """Only filenames/metadata are read here; immutable split must already exist."""
    manifest=root/config['split_manifest']
    if sha256(manifest)!=config['split_sha256']:
        raise ValueError('Missing or changed locked manifest; never create another split')
    frame=pd.read_csv(manifest)
    if frame.isna().any().any() or frame.image_id.duplicated().any() or set(frame.split)!={'train','val','test'}:
        raise ValueError('Invalid partition metadata')
    if not (frame.label==frame.diagnosis.map(dict(zip(CLASSES,range(7))))).all():
        raise ValueError('Class mapping changed')
    groups={s:set(frame.loc[frame.split==s,'lesion_id']) for s in ('train','val','test')}
    if any(groups[a]&groups[b] for a,b in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('Lesion overlap')
    metadata=root/'data/raw/HAM10000/HAM10000_metadata.csv'
    raw=pd.read_csv(metadata).set_index('image_id')[['lesion_id','dx']].rename(columns={'dx':'diagnosis'})
    if not raw.sort_index().equals(frame.set_index('image_id')[['lesion_id','diagnosis']].sort_index()):
        raise ValueError('Manifest differs from raw metadata')
    dev=frame.loc[frame.split.isin(['train','val'])].copy()
    files={}
    for folder in ('HAM10000_images_part_1','HAM10000_images_part_2'):
        directory=root/'data/raw/HAM10000'/folder
        if not directory.is_dir(): raise FileNotFoundError(directory)
        for p in directory.glob('*.jpg'):
            if p.stem in files: raise ValueError('Duplicate filename mapping')
            files[p.stem]=p
    dev['path']=[str(files[i]) for i in dev.image_id] # missing files fail, no test paths retained
    train=dev.loc[dev.split=='train'].reset_index(drop=True)
    val=dev.loc[dev.split=='val'].reset_index(drop=True)
    counts=np.asarray([(train.label==i).sum() for i in range(7)])
    if len(train)!=7009 or len(val)!=1503 or (counts==0).any():
        raise ValueError('Unexpected development support')
    weights=np.sqrt(len(train)/(7*counts));weights/=weights.mean()
    return train,val,torch.tensor(weights,dtype=torch.float32)


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
    return {f:sha256(ROOT/f) for f in ('research/train.py','research/models.py','research/experiment.py',
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


def restore_experiment(config, checkpoint):
    """Checkpoint is the authoritative committed epoch, repairing partial CSV writes."""
    from .registry import upsert
    path=ROOT/'results/structured_experiments'/config['experiment_id']
    saved=json.loads((path/'config.json').read_text(encoding='utf-8'))
    if (saved!=config or checkpoint['config']!=config or checkpoint['code_hashes']!=code_hashes()
            or checkpoint['runtime_versions']!=runtime_versions()):
        raise ValueError('Resume config or runner changed; do not silently continue another recipe')
    obj=Experiment.__new__(Experiment);obj.config=saved;obj.path=path
    obj.history=checkpoint['history'];obj.row=checkpoint['registry_row']
    obj.validation_support=pd.read_csv(ROOT/config['split_manifest']).query("split=='val'").diagnosis.value_counts().to_dict()
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


def run(config,resume=False):
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
        experiment=restore_experiment(config,checkpoint)
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
    try:
        logger.info('START %s resume=%s train=%d val=%d; test loader absent',config['experiment_id'],resume,len(train),len(val))
        progress(status='initializing',epoch=len(experiment.history),max_epochs=config['max_epochs'])
        # Resume loads trained weights; no ImageNet download on that path.
        model=ResearchClassifier(config['model'],weights=None if resume else config['weights'],
                                  attention=config['attention'],dropout=config['head_dropout']).cuda()
        optimizer=torch.optim.AdamW(optimizer_groups(model,config),betas=tuple(config['optimizer_betas']),
                                   eps=config['optimizer_eps'],weight_decay=config['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,
                                   threshold=0,threshold_mode='abs',cooldown=0,min_lr=1e-7)
        scaler=torch.amp.GradScaler('cuda')
        weights=weights.cuda();best=None;stale=0;updates=0
        if checkpoint:
            model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler']);scaler.load_state_dict(checkpoint['scaler'])
            best=checkpoint['best'];stale=checkpoint['stale'];updates=checkpoint['optimizer_updates']
            restore_rng(checkpoint['rng'])
        else:
            enum=models.get_model_weights(config['model'])[config['weights'].split('.')[-1]]
            cached=Path(torch.hub.get_dir())/'checkpoints'/enum.url.rsplit('/',1)[-1]
            write_json(experiment.path/'pretraining.json',dict(weights=config['weights'],url=enum.url,
                       sha256=sha256(cached),parameters=sum(p.numel() for p in model.parameters()),
                       metadata_sha256=sha256(ROOT/'data/raw/HAM10000/HAM10000_metadata.csv'),
                       code_hashes=code_hashes(),torch=torch.__version__,device=torch.cuda.get_device_name(0),
                       microbatch=config['batch_size'],effective_batch=config['effective_batch_size'],
                       native_attention='SE retained for EfficientNet/MobileNet; none means no added CBAM'))
        trainset=DevelopmentImages(train,config,True);valset=DevelopmentImages(val,config)
        val_loader=DataLoader(valset,batch_size=config['validation_batch_size'],shuffle=False,num_workers=config['workers'],
                              pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(config['seed']))
        for epoch in range(len(experiment.history)+1,config['max_epochs']+1):
            if stale>=config['patience']: break
            tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
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
                scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
                updates+=int(scaler.get_scale()>=scale_before)
                if step%20==0 or step+1==steps:
                    progress(status='training',epoch=epoch,step=step+1,steps=steps,max_epochs=config['max_epochs'],
                             best_epoch=best['epoch'] if best else None,best_macro_f1=best['metrics']['macro_f1'] if best else None)
                    logger.info('epoch %d/%d step %d/%d weighted_loss %.5f',epoch,config['max_epochs'],step+1,steps,numerator/denominator)
            model.eval();all_y=[];all_p=[];ids=[];vn=vd=0.
            progress(status='validating',epoch=epoch,max_epochs=config['max_epochs'])
            vt=time.perf_counter()
            with torch.inference_mode():
                for images,y,image_ids in val_loader:
                    images=images.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
                    with torch.autocast('cuda',dtype=torch.float16): logits=model(images)
                    vn+=float(weighted_numerator(logits,y,weights));vd+=float(weights[y].sum())
                    all_y.extend(y.cpu().tolist());all_p.extend(logits.float().softmax(1).cpu().tolist());ids.extend(image_ids)
            validation_ms=(time.perf_counter()-vt)*1000/len(val)
            metrics=metric_report(np.asarray(all_y),all_p,vn/vd)
            if not np.isfinite(metrics['loss']): raise FloatingPointError('Nonfinite validation loss')
            lr_before=[g['lr'] for g in optimizer.param_groups];scheduler.step(metrics['macro_f1'])
            row=dict(epoch=epoch,train_loss=numerator/denominator,train_accuracy=correct/used,train_images_used=used,
                     val_loss=metrics['loss'],val_accuracy=metrics['accuracy'],val_macro_precision=metrics['macro_precision'],
                     val_macro_recall=metrics['macro_recall'],val_macro_f1=metrics['macro_f1'],epoch_seed=epoch_seed,
                     backbone_lr=lr_before[0],head_lr=lr_before[1],optimizer_updates=updates,
                     validation_ms_per_image=validation_ms,peak_allocated_vram_mb=torch.cuda.max_memory_allocated()/1024**2,
                     epoch_seconds=time.perf_counter()-tick)
            predictions=[dict(image_id=i,true_class=CLASSES[y],predicted_class=CLASSES[int(np.argmax(p))],
                              **{f'p_{c}':float(p[j]) for j,c in enumerate(CLASSES)}) for i,y,p in zip(ids,all_y,all_p)]
            improved=best is None or metrics['macro_f1']>best['metrics']['macro_f1']
            stale=0 if improved else stale+1
            if improved:
                best=dict(epoch=epoch,metrics=metrics,predictions=predictions,
                          model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            # Commit full state first. If CSV/registry/plots fail, explicit resume repairs
            # them from this snapshot without rerunning an already committed epoch.
            payload=dict(config=config,code_hashes=code_hashes(),runtime_versions=runtime_versions(),model=model.state_dict(),optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(True),best=best,stale=stale,
                         history=experiment.history+[row],registry_row=copy.deepcopy(experiment.row),
                         optimizer_updates=updates,runtime_seconds=runtime+time.perf_counter()-start)
            atomic_checkpoint(latest,payload)
            if improved: atomic_checkpoint(ckptdir/'best.pt',dict(payload,model=best['model']))
            experiment.log_epoch(row)
            logger.info('EPOCH %d val_accuracy=%.6f macro_f1=%.6f best_epoch=%d stale=%d',epoch,metrics['accuracy'],metrics['macro_f1'],best['epoch'],stale)
        if best is None: raise RuntimeError('No committed epoch; cannot publish a completed result')
        # This also repairs a best.pt write interrupted after latest.pt committed.
        final=dict(config=config,code_hashes=code_hashes(),model=best['model'],best_epoch=best['epoch'],
                   metrics=best['metrics'],class_order=list(CLASSES),selection_metric='macro_f1')
        atomic_checkpoint(ckptdir/'best.pt',final)
        write_csv(experiment.path/'validation_predictions.csv',best['predictions'])
        write_json(experiment.path/'training_summary.json',dict(parameters=sum(p.numel() for p in model.parameters()),
                   optimizer_updates=updates,best_epoch=best['epoch'],stop_reason='patience' if stale>=config['patience'] else 'epoch_cap',
                   validation_timing_scope='End-to-end validation batch loop including loading/transfers; not pure GPU latency',
                   best_validation_ms_per_image=experiment.history[best['epoch']-1]['validation_ms_per_image']))
        experiment.complete(best['metrics'],ckptdir/'best.pt',best['epoch'],runtime+time.perf_counter()-start,
                            'Common recipe v1; validation only; selected earliest maximum macro-F1; no test loader.')
        progress(status='completed',epochs=len(experiment.history),best_epoch=best['epoch'],
                 best_accuracy=best['metrics']['accuracy'],best_macro_f1=best['metrics']['macro_f1'])
        logger.info('COMPLETED best_epoch=%d accuracy=%.6f macro_f1=%.6f',best['epoch'],best['metrics']['accuracy'],best['metrics']['macro_f1'])
    except BaseException as exc:
        experiment.fail(repr(exc));progress(status='failed',error=repr(exc),committed_epochs=len(experiment.history))
        logger.exception('FAILED; committed checkpoints preserved');raise
    finally:
        for handler in handlers: logger.removeHandler(handler);handler.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--check',action='store_true',help='Read-only config/data-path check; no model, CUDA, downloads or run outputs')
    parser.add_argument('--resume',action='store_true',help='Resume trusted latest.pt at committed epoch boundary')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'));validate_config(config)
    if args.check:
        train,val,w=development_data(config)
        print(json.dumps(dict(status='ready_pending_gpu_approval',experiment_id=config['experiment_id'],
                   config_sha256=sha256(args.config),train_images=len(train),validation_images=len(val),
                   class_weights=dict(zip(CLASSES,w.tolist())),test_loader=False,cuda_used=False,
                   checkpoints=f"checkpoints/structured/{config['experiment_id']}",
                   results=f"results/structured_experiments/{config['experiment_id']}"),indent=2))
    else:
        with run_lock(config['experiment_id']): run(config,args.resume)


if __name__=='__main__': main()
