"""One frozen-feature preflight followed by a gated, capped ConvNeXt-V2 run."""
import argparse, json, logging, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import timm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from research.common import ROOT, CLASSES, sha256, relative, write_json, write_csv
from research.models import ResearchClassifier
from research.strict_train import (DevelopmentImages, seed_worker, atomic_checkpoint, rng_state,
    restore_rng, run_lock, metric_report, evaluate_validation, runtime_versions)
from research.aggressive.core import predictions, retry_registry_upsert
from research.plots import metric_figures, training_figures, comparison_figures

RID = 's57_convnextv2_tiny_22k_exploratory_seed42'
BASE = ROOT/'results/short_screening/convnextv2_transfer_v1'
OUT = BASE/RID
CK = ROOT/'checkpoints/short_screening/convnextv2_transfer_v1'/RID
CFG = ROOT/'research/short_screening/convnextv2_transfer_v1.json'
PROBE = BASE/'s56_frozen_feature_screen'
MODEL = 'convnextv2_tiny.fcmae_ft_in22k_in1k'


def config():
    c = json.loads(CFG.read_text())
    assert c['model']==MODEL and c['max_epochs']==20 and not c['test_evaluation']
    assert c['class_order']==list(CLASSES) and c['image_size']==224
    return c


def development(c):
    manifest=ROOT/c['split_manifest'];assert sha256(manifest)==c['split_sha256']
    ids=pd.read_csv(manifest,usecols=['image_id','lesion_id','split'])
    excluded=set((ids.index[ids.split=='test']+1).tolist())
    dev=pd.read_csv(manifest,skiprows=lambda line:line in excluded)
    assert set(dev.split)=={'train','val'} and dev.image_id.is_unique
    assert dev.label.tolist()==dev.diagnosis.map(dict(zip(CLASSES,range(7)))).tolist()
    locked=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','lesion_id','split'])
    test=locked.loc[locked.split=='test']
    assert not set(dev.image_id)&set(test.image_id) and not set(dev.lesion_id)&set(test.lesion_id)
    paths={}
    for part in ['HAM10000_images_part_1','HAM10000_images_part_2']:
        for p in (ROOT/'data/raw/HAM10000'/part).glob('*.jpg'):
            assert p.stem not in paths;paths[p.stem]=p
    dev['path']=[str(paths[i]) for i in dev.image_id]
    train=dev.loc[dev.split=='train'].reset_index(drop=True);val=dev.loc[dev.split=='val'].reset_index(drop=True)
    assert len(train)==7009 and len(val)==1503
    counts=np.bincount(train.label,minlength=7);assert (counts>0).all()
    weights=np.sqrt(len(train)/(7*counts));weights/=weights.mean()
    return train,val,torch.tensor(weights,dtype=torch.float32)


def fresh(classes=7):
    # Download the exact author-published URL, rather than an unpinned HF revision.
    net=timm.create_model(MODEL,pretrained=True,num_classes=classes,drop_rate=.2,
                         drop_path_rate=0.,pretrained_cfg_overlay={'hf_hub_id':None})
    return net


def pretrained_file():
    url=timm.get_pretrained_cfg(MODEL).url
    p=Path(torch.hub.get_dir())/'checkpoints'/url.rsplit('/',1)[-1]
    assert p.is_file();return p


def fingerprints():
    return {f:sha256(ROOT/f) for f in ['research/short_screening/convnextv2_transfer.py',
        'research/short_screening/convnextv2_transfer_v1.json','research/models.py',
        'research/strict_train.py','research/common.py','research/plots.py','research/aggressive/core.py']}


def check():
    c=config();train,val,w=development(c);torch.set_num_threads(4);torch.manual_seed(42)
    net=fresh();net.train();ds=DevelopmentImages(train,c,True)
    x=torch.stack([ds[i][0] for i in range(2)]);y=torch.tensor([ds[i][1] for i in range(2)])
    z=net(x);loss=F.cross_entropy(z.float(),y,weight=w);loss.backward()
    assert z.shape==(2,7) and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    p=pretrained_file()
    report=dict(cpu_forward_backward='passed',parameters=sum(p.numel() for p in net.parameters()),train_images=len(train),val_images=len(val),
        source_url=timm.get_pretrained_cfg(MODEL).url,pretrained_file=relative(p),pretrained_sha256=sha256(p),
        class_order=list(CLASSES),test_loaded=False,config_sha256=sha256(CFG),fingerprints=fingerprints(),timm_version=timm.__version__)
    write_json(BASE/'preflight.json',report);print(json.dumps(report,indent=2))


def probe():
    if (PROBE/'summary.json').exists(): print((PROBE/'summary.json').read_text());return
    c=config();train,val,w=development(c)
    PROBE.mkdir(parents=True,exist_ok=True)
    write_json(PROBE/'PREDECLARED_PLAN.json',dict(
        control='Fresh torchvision ConvNeXt-Tiny ImageNet1k, frozen normalized pooled features',
        candidate=MODEL,head='Same StandardScaler + LogisticRegression C1 max_iter1000 on train only; sqrt class sample weights',
        transforms='Same square224 bilinear/ImageNet normalization; identity only; no test loader',
        gate='Candidate accuracy >= control+0.01 OR macro-F1 >= control+0.02 with accuracy nondecreasing',
        limit='Two pretrained feature models, one head configuration each; no head tuning',
        caveat='Frozen feature gate supports a trial, not a prediction of fine-tuned or ensemble performance'))
    torch.set_num_threads(4);torch.manual_seed(42)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(PROBE/'run.log'),logging.StreamHandler()])
    cache=ROOT/'.cache/convnextv2_transfer_v1';cache.mkdir(exist_ok=True)
    metrics={};tick=time.perf_counter()
    for name in ['control_1k','candidate_v2_22k']:
        if name=='control_1k':net=ResearchClassifier('convnext_tiny',weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1').cuda()
        else:net=fresh(classes=0).cuda()
        net.eval();features=[]
        for part,frame in [('train',train),('val',val)]:
            vectors=[];seen=[];ys=[]
            dataset=DevelopmentImages(frame,c,training=(part=='train'))
            if part=='train':
                # The guarded train loader remains a train loader; frozen probes
                # use identity transforms on both partitions, without relabeling.
                from torchvision import transforms
                dataset.transform=transforms.Compose([t for t in dataset.transform.transforms if not isinstance(t,transforms.RandomHorizontalFlip)])
            loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=2,pin_memory=True,worker_init_fn=seed_worker)
            with torch.inference_mode():
                for step,(x,y,ids) in enumerate(loader):
                    x=x.cuda(non_blocking=True)
                    with torch.autocast('cuda',dtype=torch.float16):
                        z=net.head[:3](net.features(x)) if name=='control_1k' else net(x)
                    vectors.append(z.float().cpu().numpy());seen.extend(ids);ys.extend(y.tolist())
                    if step%60==0:logging.info('%s %s features %s/%s',name,part,len(seen),len(frame))
            a=np.concatenate(vectors);assert a.shape==(len(frame),768) and np.isfinite(a).all()
            assert seen==frame.image_id.tolist() and ys==frame.label.tolist()
            np.savez_compressed(cache/f'{name}_{part}.npz',features=a,ids=np.array(seen),y=np.array(ys))
            features.append(a)
        del net;torch.cuda.empty_cache()
        classifier=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
        with threadpool_limits(limits=4):
            classifier.fit(features[0],train.label,logisticregression__sample_weight=w[train.label.to_numpy()].numpy())
            assert classifier[-1].classes_.tolist()==list(range(7))
            p=classifier.predict_proba(features[1])
        import joblib
        joblib.dump(classifier,cache/f'{name}_head.joblib')
        assert classifier[-1].n_iter_.max()<1000, 'Frozen probe head did not converge; gate invalid'
        y=val.label.to_numpy();m=metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))
        target=PROBE/name;write_json(target/'validation_metrics.json',m);pred=predictions(val.image_id,y,p)
        write_csv(target/'validation_predictions.csv',pred)
        write_csv(target/'validation_probabilities.csv',pd.DataFrame(pred)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
        metric_figures(m,target/'figures',f'S56 frozen feature probe {name} | exploratory validation')
        metrics[name]=m;logging.info('%s accuracy=%.6f macroF1=%.6f',name,m['accuracy'],m['macro_f1'])
    a=metrics['control_1k'];b=metrics['candidate_v2_22k']
    passed=b['accuracy']>=a['accuracy']+.01-1e-12 or (b['macro_f1']>=a['macro_f1']+.02-1e-12 and b['accuracy']>=a['accuracy'])
    comparison_figures([dict(display_name=k,accuracy=m['accuracy'],macro_f1=m['macro_f1']) for k,m in metrics.items()],PROBE/'comparison_figures','Matched frozen-feature transfer screen')
    summary=dict(status='completed',gate_passed=bool(passed),metrics=metrics,runtime_seconds=time.perf_counter()-tick,test_loaded=False,
        training_epochs=0,feature_extraction_gpu=True,preflight_sha256=sha256(BASE/'preflight.json'),config_sha256=sha256(CFG))
    write_json(PROBE/'summary.json',summary)
    for name,m in metrics.items():
        retry_registry_upsert(dict(experiment_id='s56_frozen_'+name+'_exploratory_seed42',era='structured',record_kind='frozen_feature_probe',
            phase='post_test_exploratory_development',protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],
            split_sha256=c['split_sha256'],model=MODEL if name=='candidate_v2_22k' else 'convnext_tiny',method='Frozen pretrained features; fixed weighted logistic regression C1',
            epochs=0,seed=42,status='completed',decision='gate_passed' if passed else 'gate_failed',metrics_path=relative(PROBE/name/'validation_metrics.json'),
            plots_dir=relative(PROBE/name/'figures'),notes='Train-only head fitting; exploratory validation screening; no full-network training/test.',
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    print(json.dumps(summary,indent=2))


def run(resume=False):
    c=config();screen=json.loads((PROBE/'summary.json').read_text());assert screen['gate_passed']
    assert screen['config_sha256']==sha256(CFG)
    preflight=json.loads((BASE/'preflight.json').read_text())
    assert preflight['fingerprints']==fingerprints() and preflight['pretrained_sha256']==sha256(pretrained_file())
    if (OUT/'summary.json').exists():print('Completed; preserved.');return
    OUT.mkdir(parents=True,exist_ok=resume);CK.mkdir(parents=True,exist_ok=resume)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    train,val,w=development(c);latest=CK/'latest.pt'
    versions=dict(**runtime_versions(),timm=timm.__version__)
    cp=torch.load(latest,map_location='cpu',weights_only=False) if resume else None
    if cp:assert cp['config']==c and cp['fingerprints']==fingerprints() and cp['runtime_versions']==versions
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'train.log'),logging.StreamHandler()])
    net=(timm.create_model(MODEL,pretrained=False,num_classes=7,drop_rate=.2,drop_path_rate=0.) if cp else fresh()).cuda()
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
        row=dict(experiment_id=RID,era='structured',record_kind='training_run',phase='post_test_exploratory_development',protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],method=c['question'],model=MODEL,pretrained_weights=MODEL,image_size=224,seed=42,loss='weighted_cross_entropy',augmentation=c['augmentation'],epochs=len(history),config_path=relative(OUT/'config.json'),history_path=relative(OUT/'history.csv'),status=status,notes='Frozen-feature gate passed; hard20 maximum; no test or automatic next training; FP32 validation.')
        if best:row.update({k:best['metrics'][k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']},best_epoch=best['epoch'],checkpoint=relative(CK/'best.pt'),metrics_path=relative(OUT/'validation_metrics.json'),plots_dir=relative(OUT/'figures'))
        retry_registry_upsert(row);write_json(OUT/'record.json',row)
    write_json(OUT/'config.json',c)
    write_json(OUT/'verification.json',dict(preflight=json.loads((BASE/'preflight.json').read_text()),class_order=list(CLASSES),train_images=len(train),val_images=len(val),test_loaded=False,fingerprints=fingerprints(),runtime_versions=versions,pretrained_sha256=sha256(pretrained_file())))
    atomic_checkpoint(latest,payload());register('running')
    logging.info('START %s resume=%s committed_epoch=%s; hard20; FP32 validation; latest checkpoint committed',RID,resume,len(history))
    write_json(OUT/'progress.json',dict(status='initialized',pid=os.getpid(),committed_epoch=len(history),checkpoint_created=True))
    trainset=DevelopmentImages(train,c,True)
    vloader=DataLoader(DevelopmentImages(val,c),batch_size=16,shuffle=False,num_workers=2,pin_memory=True,worker_init_fn=seed_worker)
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
                if finite:torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True)
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
            if epoch>=15 and stale>=6 and optimizer.param_groups[0]['lr']<3e-5:
                stop_reason='meaningful_plateau_after_min15_and_lr_reduction';break
        for suffix,selected in [('',best),('_macro_f1',f1best)]:
            write_csv(OUT/f'validation_probabilities{suffix}.csv',pd.DataFrame(selected['predictions'])[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
            metric_figures(selected['metrics'],OUT/f'figures{suffix}',f'S57 ConvNeXt-V2 Tiny | exploratory validation{suffix}')
        training_figures(pd.DataFrame(history),OUT/'figures','S57 ConvNeXt-V2 Tiny training',selection_metric='accuracy')
        comparison_figures([dict(display_name='S06 Tiny 1k',accuracy=.9174983366600133,macro_f1=.862830691053104),dict(display_name='S18 Small 1k',accuracy=.9221556886227545,macro_f1=.8549484177398866),dict(display_name='S57 V2 Tiny 22k',accuracy=best['metrics']['accuracy'],macro_f1=best['metrics']['macro_f1'])],OUT/'comparison_figures','Standalone exploratory transfer comparison')
        register('completed')
        write_json(OUT/'summary.json',dict(status='completed',epochs=len(history),best_epoch=best['epoch'],best_macro_f1_epoch=f1best['epoch'],best_accuracy=best['metrics']['accuracy'],best_macro_f1=f1best['metrics']['macro_f1'],final_metrics=m,stop_reason=stop_reason,runtime_seconds=prior_runtime+time.perf_counter()-start,test_loaded=False,auto_next_experiment=False))
        write_json(OUT/'progress.json',dict(status='completed',pid=os.getpid(),committed_epoch=len(history),stop_reason=stop_reason))
    except Exception as exc:
        write_json(OUT/'failure.json',dict(error=repr(exc),committed_epoch=len(history),latest_checkpoint=relative(latest),test_loaded=False));register('failed');raise


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--check',action='store_true');a.add_argument('--probe',action='store_true');a.add_argument('--run',action='store_true');a.add_argument('--resume',action='store_true');args=a.parse_args()
    assert sum([args.check,args.probe,args.run])==1 and (not args.resume or args.run)
    with run_lock(RID):
        if args.check:check()
        elif args.probe:probe()
        else:run(args.resume)
