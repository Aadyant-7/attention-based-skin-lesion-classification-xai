"""Isolated S95 ConvNeXt-Tiny weight-decay experiment; historical code remains unchanged.

Training-loop provenance: the verified S89 full-window loop, using ResearchClassifier
and an exact20-epoch budget. No automatic extension or next experiment.
"""
import argparse, json, logging, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision import models
from torch.nn import functional as F
from torch.utils.data import DataLoader
from research.common import ROOT, CLASSES, sha256, relative, write_json, write_csv
from research.models import ResearchClassifier
from research.short_screening.supervised22k_transfer import development
from research.strict_train import (DevelopmentImages, seed_worker, atomic_checkpoint, rng_state,
    restore_rng, run_lock, metric_report, evaluate_validation, runtime_versions)
from research.aggressive.core import predictions, retry_registry_upsert
from research.plots import metric_figures, training_figures, comparison_figures

RID='s95_convnext_tiny_wd005_exploratory_seed42'
MODEL='convnext_tiny'
BASE=ROOT/'results/short_screening/convnext_regularization_v1'
OUT=BASE/RID
CK=ROOT/'checkpoints/short_screening/convnext_regularization_v1'/RID
CFG=ROOT/'research/short_screening/s95_convnext_regularization_v1.json'


def config():
    c=json.loads(CFG.read_text())
    assert c['model']==MODEL and c['max_epochs']==20 and c['minimum_epochs']==20
    assert c['weights']=='ConvNeXt_Tiny_Weights.IMAGENET1K_V1'
    assert not c['test_evaluation'] and not c['auto_next_experiment']
    assert c['class_order']==list(CLASSES) and c['image_size']==224
    expected=dict(batch_size=16,validation_batch_size=16,effective_batch_size=32,
        gradient_accumulation=2,weight_decay=.05,head_dropout=.2,workers=2,
        loss='weighted_cross_entropy',optimizer='AdamW',validation_precision='fp32',seed=42)
    assert all(c[k]==v for k,v in expected.items()), 'Fixed runner recipe does not match config'
    assert c['learning_rate']==dict(backbone=3e-5,head=1e-4)
    assert c['attention']=='none' and c['weight_decay']==.05
    assert c['gradient_clip_norm'] is None
    assert c['experiment_id']==RID and c['initial_freeze_epochs']==0
    assert c['optimizer_betas']==[.9,.999] and c['optimizer_eps']==1e-8
    reference=json.loads((ROOT/'results/structured_experiments/s06_convnext_tiny_none_exploratory_seed42/config.json').read_text())
    matched=['model','weights','attention','seed','image_size','split_manifest','split_sha256','class_order',
             'head_dropout','augmentation','imbalance','loss','loss_reduction','learning_rate',
             'optimizer','optimizer_betas','optimizer_eps','batch_size','validation_batch_size',
             'effective_batch_size','gradient_accumulation','normalization_mean','normalization_std','max_epochs','scheduler']
    assert all(c[k]==reference[k] for k in matched), 'Unplanned S06 recipe change'
    return c


def fresh(pretrained=True):
    return ResearchClassifier(MODEL,weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1' if pretrained else None,
                              attention='none',dropout=.2)


def pretrained_file():
    return Path(torch.hub.get_dir())/'checkpoints'/models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url.rsplit('/',1)[-1]


def fingerprints():
    return {f:sha256(ROOT/f) for f in ['research/short_screening/s95_convnext_regularization.py',
        'research/short_screening/s95_convnext_regularization_v1.json','research/short_screening/supervised22k_transfer.py',
        'results/structured_experiments/s06_convnext_tiny_none_exploratory_seed42/config.json',
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
    head={id(p) for p in net.head[-1].parameters()}
    groups=[dict(params=[p for p in net.parameters() if id(p) not in head],lr=3e-5),
            dict(params=list(net.head[-1].parameters()),lr=1e-4)]
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=.05)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in net.parameters())
    assert len({id(p) for g in optimizer.param_groups for p in g['params']})==len(list(net.parameters()))
    assert all(g['weight_decay']==.05 for g in optimizer.param_groups)
    # Verify actual CPU optimizer/checkpoint/RNG restoration, never CUDA.
    import tempfile
    with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as scratch:
        saved=Path(scratch)/'resume.pt';rng=rng_state(False);expected=torch.rand(4)
        atomic_checkpoint(saved,dict(model=net.state_dict(),optimizer=optimizer.state_dict(),rng=rng,config=c))
        cp=torch.load(saved,map_location='cpu',weights_only=False)
        restored=fresh(pretrained=False);restored.load_state_dict(cp['model'],strict=True)
        rh={id(p) for p in restored.head[-1].parameters()}
        rg=[dict(params=[p for p in restored.parameters() if id(p) not in rh],lr=3e-5),dict(params=list(restored.head[-1].parameters()),lr=1e-4)]
        resumed=torch.optim.AdamW(rg,weight_decay=.05);resumed.load_state_dict(cp['optimizer'])
        for a,b in zip(net.state_dict().values(),restored.state_dict().values()):assert torch.equal(a,b)
        assert resumed.state_dict()['param_groups']==optimizer.state_dict()['param_groups']
        for key,state in optimizer.state_dict()['state'].items():
            for field,a in state.items():
                b=resumed.state_dict()['state'][key][field]
                assert torch.equal(a,b) if torch.is_tensor(a) else a==b
        restore_rng(cp['rng']);actual=torch.rand(4);assert torch.equal(actual,expected)
    rejected=val.iloc[:1].copy();rejected['split']='test'
    try:DevelopmentImages(rejected,c)
    except ValueError:pass
    else:raise AssertionError('Test partition accepted by development dataset')
    report=dict(cpu_forward_backward='passed',parameters=sum(p.numel() for p in net.parameters()),
        train_images=len(train),val_images=len(val),source_url=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url,
        pretrained_file=relative(path),pretrained_sha256=sha256(path),class_order=list(CLASSES),
        test_loaded=False,gpu_used=False,config_sha256=sha256(CFG),fingerprints=fingerprints(),
        optimizer_step_and_group_decay='passed',cpu_checkpoint_optimizer_rng_recovery='passed',
        test_dataset_guard='passed',maximum_epochs=20,
        warning='Two-sample CPU safety check is not an accuracy screen or evidence of future ensemble gain.')
    write_json(BASE/'preflight.json',report);print(json.dumps(report,indent=2))


def run(resume=False,stop_after_epoch=20):
    c=config();assert stop_after_epoch==20, 'One authorized20-epoch window only'
    preflight=json.loads((BASE/'preflight.json').read_text())
    assert preflight['fingerprints']==fingerprints() and preflight['pretrained_sha256']==sha256(pretrained_file())
    if (OUT/'summary.json').exists():
        saved=json.loads((OUT/'summary.json').read_text())
        if saved['status']=='completed' or not resume or saved['epochs']>=stop_after_epoch:
            print('Completed S95 preserved; no restart or extension.');return
    OUT.mkdir(parents=True,exist_ok=resume);CK.mkdir(parents=True,exist_ok=resume)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    train,val,w=development(c);latest=CK/'latest.pt'
    versions=runtime_versions()
    cp=torch.load(latest,map_location='cpu',weights_only=False) if resume else None
    if cp:assert cp['config']==c and cp['fingerprints']==fingerprints() and cp['runtime_versions']==versions
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'train.log'),logging.StreamHandler()])
    net=(fresh(pretrained=False) if cp else fresh()).cuda()
    head={id(p) for p in net.head[-1].parameters()}
    groups=[dict(params=[p for p in net.parameters() if id(p) not in head],lr=3e-5),dict(params=list(net.head[-1].parameters()),lr=1e-4)]
    optimizer=torch.optim.AdamW(groups,betas=tuple(c['optimizer_betas']),eps=c['optimizer_eps'],weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,threshold=0.,threshold_mode='abs',min_lr=1e-7)
    scaler=torch.amp.GradScaler('cuda');w=w.cuda();history=[];best=f1best=None;updates=0;prior_runtime=0
    meaningful=dict(accuracy=None,macro_f1=None,last_accuracy_epoch=0,last_f1_epoch=0)
    if cp:
        net.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler']);scaler.load_state_dict(cp['scaler'])
        history=cp['history'];best=cp['best'];f1best=cp['f1best'];updates=cp['updates'];meaningful=cp['meaningful'];prior_runtime=cp['runtime_seconds'];restore_rng(cp['rng']);del cp
    start=time.perf_counter();stop_reason='hard20_epoch_cap'
    def payload():return dict(config=c,fingerprints=fingerprints(),runtime_versions=versions,model=net.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),history=history,best=best,f1best=f1best,rng=rng_state(True),meaningful=meaningful,updates=updates,runtime_seconds=prior_runtime+time.perf_counter()-start)
    def register(status):
        row=dict(experiment_id=RID,era='structured',record_kind='training_run',phase='post_test_exploratory_development',protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],method=c['question'],model=MODEL,pretrained_weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1',image_size=224,seed=42,loss='weighted_cross_entropy',optimizer='AdamW',backbone_lr=3e-5,head_lr=1e-4,batch_size=16,augmentation=c['augmentation'],epochs=len(history),config_path=relative(OUT/'config.json'),history_path=relative(OUT/'history.csv'),status=status,notes=c['comparability']+' No test or automatic next training; FP32 validation; exact20 window.')
        if best:row.update({k:best['metrics'][k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']},best_epoch=best['epoch'],checkpoint=relative(CK/'best.pt'),metrics_path=relative(OUT/'validation_metrics.json'),plots_dir=relative(OUT/'figures'))
        retry_registry_upsert(row);write_json(OUT/'record.json',row)
    write_json(OUT/'config.json',c)
    write_json(OUT/'verification.json',dict(preflight=json.loads((BASE/'preflight.json').read_text()),class_order=list(CLASSES),train_images=len(train),val_images=len(val),test_loaded=False,fingerprints=fingerprints(),runtime_versions=versions,pretrained_sha256=sha256(pretrained_file())))
    atomic_checkpoint(latest,payload());register('running')
    logging.info('START %s resume=%s committed_epoch=%s; stop_boundary=%s; hard20; FP32 validation; latest checkpoint committed',RID,resume,len(history),stop_after_epoch)
    write_json(OUT/'progress.json',dict(status='initialized',pid=os.getpid(),committed_epoch=len(history),checkpoint_created=True))
    trainset=DevelopmentImages(train,c,True)
    vloader=DataLoader(DevelopmentImages(val,c),batch_size=16,shuffle=False,num_workers=2,pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42))
    try:
        for epoch in range(len(history)+1,21):
            tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();net.train()
            loader=DataLoader(trainset,batch_size=16,shuffle=True,drop_last=True,num_workers=2,pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42+epoch))
            assert len(loader)%2==0
            it=iter(loader);total_num=total_den=0.;correct=used=skipped=0
            for step in range(len(loader)//2):
                pair=[next(it),next(it)];target=torch.cat([b[1] for b in pair]).cuda();den=w[target].sum();optimizer.zero_grad(set_to_none=True)
                for x,y,_ in pair:
                    x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
                    with torch.autocast('cuda',dtype=torch.float16):z=net(x)
                    num=F.cross_entropy(z.float(),y,weight=w,reduction='sum')
                    if not torch.isfinite(num):raise FloatingPointError('Nonfinite training logits/loss; latest epoch preserved')
                    scaler.scale(num/den).backward();total_num+=float(num.detach());total_den+=float(w[y].sum());correct+=int((z.argmax(1)==y).sum());used+=len(y)
                scaler.unscale_(optimizer)
                finite=all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
                # Preserve S06's unclipped AdamW updates. GradScaler skips nonfinite gradients.
                if not finite:logging.warning('Nonfinite AMP gradients: preserving scaler backoff; optimizer update skipped')
                old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();accepted=scaler.get_scale()>=old_scale;updates+=int(accepted);skipped+=int(not accepted)
                if step==0 or (step+1)%50==0:
                    logging.info('epoch %s/20 step %s/%s train_loss=%.5f accepted_updates=%s',epoch,step+1,len(loader)//2,total_num/total_den,updates)
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
        for suffix,selected in [('',best),('_macro_f1',f1best)]:
            write_csv(OUT/f'validation_probabilities{suffix}.csv',pd.DataFrame(selected['predictions'])[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
            metric_figures(selected['metrics'],OUT/f'figures{suffix}',f'S95 ConvNeXt-Tiny decay0.05 | exploratory validation{suffix}')
        write_csv(OUT/'validation_probabilities_latest.csv',pd.DataFrame(pred)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
        metric_figures(m,OUT/'figures_latest','S95 ConvNeXt-Tiny decay0.05 final committed epoch | exploratory validation')
        training_figures(pd.DataFrame(history),OUT/'figures','S95 ConvNeXt-Tiny decay0.05 training',selection_metric='accuracy')
        reference=json.loads((ROOT/'results/structured_experiments/s06_convnext_tiny_none_exploratory_seed42/validation_metrics.json').read_text())
        comparison_figures([dict(display_name='S06 original decay0.0001 (historical AMP)',accuracy=reference['accuracy'],macro_f1=reference['macro_f1']),dict(display_name='S95 ConvNeXt-Tiny decay0.05 (FP32 validation)',accuracy=best['metrics']['accuracy'],macro_f1=best['metrics']['macro_f1'])],OUT/'comparison_figures','Controlled regularization development | exploratory validation')
        stage_status='completed'
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
    parser.add_argument('--stop-after-epoch',type=int,choices=[20],default=20)
    args=parser.parse_args()
    assert sum([args.check,args.run])==1
    assert not args.download or args.check
    assert not args.resume or args.run
    with run_lock(RID):
        if args.check:check(args.download)
        else:run(args.resume,args.stop_after_epoch)
