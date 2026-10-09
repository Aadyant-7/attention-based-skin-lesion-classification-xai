"""Isolated DeiT III Small screening runner; no changes to historical/frozen model code.

Training-loop provenance: supervised22k_transfer.run, adapted for timm DeiT III
and a mandatory epoch15 pilot boundary. Defaults never extend past that boundary.
"""
import argparse, json, logging, os, random, time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import timm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from research.common import ROOT, CLASSES, sha256, relative, write_json, write_csv
from research.short_screening.supervised22k_transfer import development
from research.strict_train import (DevelopmentImages, seed_worker, atomic_checkpoint, rng_state,
    restore_rng, run_lock, metric_report, evaluate_validation, runtime_versions)
from research.aggressive.core import predictions, retry_registry_upsert
from research.plots import metric_figures, training_figures, comparison_figures

RID='s90_deit3_small_none_exploratory_seed42'
MODEL='deit3_small_patch16_224.fb_in22k_ft_in1k'
BASE=ROOT/'results/short_screening/deit3_transfer_v1'
OUT=BASE/RID
CK=ROOT/'checkpoints/short_screening/deit3_transfer_v1'/RID
CFG=ROOT/'research/short_screening/s90_deit3_transfer_v1.json'


def config():
    c=json.loads(CFG.read_text())
    assert c['model']==MODEL and c['max_epochs']==20 and c['minimum_epochs']==15
    assert c['weights']==MODEL
    assert not c['test_evaluation'] and not c['auto_next_experiment']
    assert c['class_order']==list(CLASSES) and c['image_size']==224
    expected=dict(batch_size=8,validation_batch_size=8,effective_batch_size=32,
        gradient_accumulation=4,weight_decay=1e-4,head_dropout=.2,workers=2,
        loss='weighted_cross_entropy',optimizer='AdamW',validation_precision='fp32',seed=42)
    assert all(c[k]==v for k,v in expected.items()), 'Fixed runner recipe does not match config'
    assert c['learning_rate']==dict(backbone=3e-5,head=1e-4)
    assert c['gpu_memory_fraction']==.4 and c['minimum_free_vram_mb']==3072 and c['drop_path_rate']==.1
    return c


def fresh(pretrained=True):
    net=timm.create_model(MODEL,pretrained=pretrained,num_classes=1000,drop_rate=0.,
        drop_path_rate=.1,pretrained_cfg_overlay={'hf_hub_id':None})
    features=net.head.in_features
    net.head=nn.Sequential(OrderedDict(dropout=nn.Dropout(.2),fc=nn.Linear(features,7)))
    return net


def pretrained_file():
    return Path(torch.hub.get_dir())/'checkpoints'/timm.get_pretrained_cfg(MODEL).url.rsplit('/',1)[-1]


def fingerprints():
    return {f:sha256(ROOT/f) for f in ['research/short_screening/s90_deit3_transfer.py',
        'research/short_screening/s90_deit3_transfer_v1.json','research/short_screening/supervised22k_transfer.py',
        'research/strict_train.py','research/models.py','research/common.py','research/plots.py','research/aggressive/core.py']}


def check(download=False):
    c=config();train,val,w=development(c);torch.set_num_threads(4);torch.manual_seed(42)
    path=pretrained_file()
    if not path.is_file() and not download:
        raise FileNotFoundError('Exact pretrained weights are not cached; --check --download authorizes the official CPU-only download')
    net=fresh();net.train();ds=DevelopmentImages(train,c,True)
    x=torch.stack([ds[i][0] for i in range(2)]);y=torch.tensor([ds[i][1] for i in range(2)])
    z=net(x);loss=F.cross_entropy(z.float(),y,weight=w);loss.backward()
    assert z.shape==(2,7) and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    net.eval()
    with torch.inference_mode():p=net(x).softmax(1)
    assert torch.isfinite(p).all() and torch.allclose(p.sum(1),torch.ones(2),atol=1e-6)
    report=dict(cpu_forward_backward='passed',parameters=sum(p.numel() for p in net.parameters()),
        train_images=len(train),val_images=len(val),source_url=timm.get_pretrained_cfg(MODEL).url,
        pretrained_file=relative(path),pretrained_sha256=sha256(path),class_order=list(CLASSES),
        test_loaded=False,gpu_used=False,config_sha256=sha256(CFG),fingerprints=fingerprints(),
        warning='Two-sample CPU safety check is not an accuracy screen or evidence of future ensemble gain.')
    write_json(BASE/'preflight.json',report);print(json.dumps(report,indent=2))


def run(resume=False,stop_after_epoch=15):
    c=config();assert stop_after_epoch in (15,20)
    assert resume or stop_after_epoch==15, 'Fresh launch is the bounded 15-epoch pilot only'
    preflight=json.loads((BASE/'preflight.json').read_text())
    assert preflight['fingerprints']==fingerprints() and preflight['pretrained_sha256']==sha256(pretrained_file())
    if (OUT/'summary.json').exists():
        saved=json.loads((OUT/'summary.json').read_text())
        if saved['status']=='completed' or not resume or saved['epochs']>=stop_after_epoch:
            print('Saved stage preserved; increasing the pilot boundary requires explicit --resume --stop-after-epoch 20 and approval.');return
    OUT.mkdir(parents=True,exist_ok=resume);CK.mkdir(parents=True,exist_ok=resume)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    train,val,w=development(c);latest=CK/'latest.pt'
    free,total=torch.cuda.mem_get_info()
    if free<3072*2**20:raise RuntimeError('Insufficient free VRAM for the authorized concurrent pilot; existing run untouched')
    torch.cuda.set_per_process_memory_fraction(.4)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'train.log'),logging.StreamHandler()])
    logging.info('Concurrent memory preflight: free=%.1fMiB total=%.1fMiB; per-process allocation cap40%%',free/2**20,total/2**20)
    write_json(OUT/'gpu_resource_preflight.json',dict(free_vram_mb=free/2**20,total_vram_mb=total/2**20,microbatch=8,accumulation=4,effective_batch_size=32,per_process_memory_fraction=.4,other_running_model_unchanged=True))
    versions=dict(**runtime_versions(),timm=timm.__version__)
    cp=torch.load(latest,map_location='cpu',weights_only=False) if resume else None
    if cp:assert cp['config']==c and cp['fingerprints']==fingerprints() and cp['runtime_versions']==versions
    net=(fresh(pretrained=False) if cp else fresh()).cuda()
    head={id(p) for p in net.head.fc.parameters()}
    groups=[dict(params=[p for p in net.parameters() if id(p) not in head],lr=3e-5),dict(params=list(net.head.fc.parameters()),lr=1e-4)]
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,threshold=.002,threshold_mode='abs',min_lr=1e-7)
    scaler=torch.amp.GradScaler('cuda');w=w.cuda();history=[];best=f1best=None;updates=0;prior_runtime=0
    meaningful=dict(accuracy=None,macro_f1=None,last_accuracy_epoch=0,last_f1_epoch=0)
    if cp:
        net.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler']);scaler.load_state_dict(cp['scaler'])
        history=cp['history'];best=cp['best'];f1best=cp['f1best'];updates=cp['updates'];meaningful=cp['meaningful'];prior_runtime=cp['runtime_seconds'];restore_rng(cp['rng']);del cp
    start=time.perf_counter();stop_reason='hard20_epoch_cap'
    def payload():return dict(config=c,fingerprints=fingerprints(),runtime_versions=versions,model=net.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),history=history,best=best,f1best=f1best,rng=rng_state(True),meaningful=meaningful,updates=updates,runtime_seconds=prior_runtime+time.perf_counter()-start)
    def register(status):
        row=dict(experiment_id=RID,era='structured',record_kind='training_run',phase='post_test_exploratory_development',protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],method=c['question'],model=MODEL,pretrained_weights=MODEL,batch_size=8,image_size=224,seed=42,loss='weighted_cross_entropy',augmentation=c['augmentation'],epochs=len(history),config_path=relative(OUT/'config.json'),history_path=relative(OUT/'history.csv'),status=status,notes='CPU safety preflight; concurrent microbatch8/accumulation4, VRAM cap40%; mandatory epoch15 pilot boundary, hard20 maximum; no test or automatic next training; FP32 validation.')
        if best:row.update({k:best['metrics'][k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']},best_epoch=best['epoch'],checkpoint=relative(CK/'best.pt'),metrics_path=relative(OUT/'validation_metrics.json'),plots_dir=relative(OUT/'figures'))
        retry_registry_upsert(row);write_json(OUT/'record.json',row)
    write_json(OUT/'config.json',c)
    write_json(OUT/'verification.json',dict(preflight=json.loads((BASE/'preflight.json').read_text()),class_order=list(CLASSES),train_images=len(train),val_images=len(val),test_loaded=False,fingerprints=fingerprints(),runtime_versions=versions,pretrained_sha256=sha256(pretrained_file())))
    atomic_checkpoint(latest,payload());register('running')
    logging.info('START %s resume=%s committed_epoch=%s; stop_boundary=%s; hard20; FP32 validation; latest checkpoint committed',RID,resume,len(history),stop_after_epoch)
    write_json(OUT/'progress.json',dict(status='initialized',pid=os.getpid(),committed_epoch=len(history),checkpoint_created=True))
    trainset=DevelopmentImages(train,c,True)
    vloader=DataLoader(DevelopmentImages(val,c),batch_size=8,shuffle=False,num_workers=2,pin_memory=True,worker_init_fn=seed_worker)
    try:
        for epoch in range(len(history)+1,21):
            tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();net.train()
            loader=DataLoader(trainset,batch_size=8,shuffle=True,drop_last=True,num_workers=2,pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42+epoch))
            assert len(loader)%4==0
            it=iter(loader);total_num=total_den=0.;correct=used=skipped=0
            for step in range(len(loader)//4):
                pair=[next(it) for _ in range(4)];target=torch.cat([b[1] for b in pair]).cuda();den=w[target].sum();optimizer.zero_grad(set_to_none=True)
                for x,y,_ in pair:
                    x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
                    with torch.autocast('cuda',dtype=torch.float16):z=net(x)
                    num=F.cross_entropy(z.float(),y,weight=w,reduction='sum')
                    if not torch.isfinite(num):raise FloatingPointError('Nonfinite training logits/loss; latest epoch preserved')
                    scaler.scale(num/den).backward();total_num+=float(num.detach());total_den+=float(w[y].sum());correct+=int((z.argmax(1)==y).sum());used+=len(y)
                scaler.unscale_(optimizer)
                finite=all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
                if finite:torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True)
                old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();accepted=scaler.get_scale()>=old_scale;updates+=int(accepted);skipped+=int(not accepted)
                if step==0 or (step+1)%50==0:
                    logging.info('epoch %s/20 step %s/%s train_loss=%.5f accepted_updates=%s',epoch,step+1,len(loader)//4,total_num/total_den,updates)
                    write_json(OUT/'progress.json',dict(status='training',pid=os.getpid(),epoch=epoch,step=step+1,committed_epoch=len(history),checkpoint_created=True,first_update_confirmed=updates>0))
            ys,ps,ids,loss,precision=evaluate_validation(net,vloader,w,c,OUT,epoch)
            assert ids==val.image_id.tolist() and precision=='fp32'
            m=metric_report(np.asarray(ys),ps,loss);pred=predictions(ids,ys,ps)
            lrs=[g['lr'] for g in optimizer.param_groups];scheduler.step(m['accuracy'])
            state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            chosen=dict(epoch=epoch,metrics=m,predictions=pred,model=state)
            for field,filename in [('accuracy','best.pt'),('macro_f1','best_macro_f1.pt')]:
                previous=best if field=='accuracy' else f1best
                if previous is None or m[field]>previous['metrics'][field]:
                    if field=='accuracy':best=chosen
                    else:f1best=chosen
                    atomic_checkpoint(CK/filename,dict(config=c,**chosen,fingerprints=fingerprints()))
            for key,delta in [('accuracy',.002),('macro_f1',.003)]:
                if meaningful[key] is None or m[key]-meaningful[key]>=delta-1e-12:
                    meaningful[key]=m[key];meaningful['last_accuracy_epoch' if key=='accuracy' else 'last_f1_epoch']=epoch
            stale=epoch-max(meaningful['last_accuracy_epoch'],meaningful['last_f1_epoch'])
            row=dict(epoch=epoch,train_loss=total_num/total_den,train_accuracy=correct/used,val_loss=loss,val_accuracy=m['accuracy'],val_macro_precision=m['macro_precision'],val_macro_recall=m['macro_recall'],val_macro_f1=m['macro_f1'],backbone_lr=lrs[0],head_lr=lrs[1],backbone_lr_after=optimizer.param_groups[0]['lr'],head_lr_after=optimizer.param_groups[1]['lr'],meaningful_stale=stale,optimizer_updates=updates,skipped_updates=skipped,peak_allocated_vram_mb=torch.cuda.max_memory_allocated()/2**20,epoch_seconds=time.perf_counter()-tick,validation_precision=precision)
            history.append(row);atomic_checkpoint(latest,payload());write_csv(OUT/'history.csv',history)
            write_csv(OUT/'lr_history.csv',[{k:r[k] for k in ['epoch','backbone_lr','head_lr','backbone_lr_after','head_lr_after']} for r in history])
            write_json(OUT/'validation_metrics_latest.json',m);write_csv(OUT/'validation_predictions_latest.csv',pred)
            for suffix,selected in [('',best),('_macro_f1',f1best)]:
                write_json(OUT/f'validation_metrics{suffix}.json',selected['metrics']);write_csv(OUT/f'validation_predictions{suffix}.csv',selected['predictions'])
            register('running');logging.info('EPOCH %s accuracy=%.6f macroF1=%.6f best_accuracy_epoch=%s meaningful_stale=%s',epoch,m['accuracy'],m['macro_f1'],best['epoch'],stale)
            write_json(OUT/'progress.json',dict(status='epoch_committed',pid=os.getpid(),committed_epoch=epoch,checkpoint_created=True,first_update_confirmed=updates>0))
            if epoch==stop_after_epoch and epoch<20:
                stop_reason='authorized_epoch15_pilot_boundary';break
            if epoch>=15 and stale>=6 and optimizer.param_groups[0]['lr']<3e-5:
                stop_reason='meaningful_plateau_after_min15_and_lr_reduction';break
        for suffix,selected in [('',best),('_macro_f1',f1best)]:
            write_csv(OUT/f'validation_probabilities{suffix}.csv',pd.DataFrame(selected['predictions'])[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
            metric_figures(selected['metrics'],OUT/f'figures{suffix}',f'S90 DeiT III Small | exploratory validation{suffix}')
        write_csv(OUT/'validation_probabilities_latest.csv',pd.DataFrame(pred)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
        metric_figures(m,OUT/'figures_latest','S90 DeiT III Small final committed epoch | exploratory validation')
        training_figures(pd.DataFrame(history),OUT/'figures','S90 DeiT III Small training',selection_metric='accuracy')
        comparison_figures([dict(display_name='S06 Tiny 1k',accuracy=.9174983366600133,macro_f1=.862830691053104),dict(display_name='S18 Small 1k',accuracy=.9221556886227545,macro_f1=.8549484177398866),dict(display_name='S90 DeiT III Small',accuracy=best['metrics']['accuracy'],macro_f1=best['metrics']['macro_f1'])],OUT/'comparison_figures','Standalone exploratory transfer comparison')
        stage_status='pilot_complete' if stop_reason=='authorized_epoch15_pilot_boundary' else 'completed'
        register(stage_status)
        write_json(OUT/'summary.json',dict(status=stage_status,epochs=len(history),best_epoch=best['epoch'],best_macro_f1_epoch=f1best['epoch'],best_accuracy=best['metrics']['accuracy'],best_macro_f1=f1best['metrics']['macro_f1'],final_metrics=m,stop_reason=stop_reason,runtime_seconds=prior_runtime+time.perf_counter()-start,test_loaded=False,auto_next_experiment=False))
        write_json(OUT/'progress.json',dict(status=stage_status,pid=os.getpid(),committed_epoch=len(history),stop_reason=stop_reason))
    except Exception as exc:
        write_json(OUT/'failure.json',dict(error=repr(exc),committed_epoch=len(history),latest_checkpoint=relative(latest),test_loaded=False));register('failed');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int,choices=[15,20],default=15)
    args=parser.parse_args()
    assert sum([args.check,args.run])==1
    assert not args.download or args.check
    assert not args.resume or args.run
    with run_lock(RID):
        if args.check:check(args.download)
        else:run(args.resume,args.stop_after_epoch)
