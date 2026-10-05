"""One fresh-ImageNet Mixup ablation, hard20 cap, development only."""
import argparse, copy, json, logging, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from research.common import ROOT, CLASSES, sha256, write_json, write_csv, relative
from research.models import ResearchClassifier
from research.strict_protocol import development_data, partition_metadata
from research.strict_train import (DevelopmentImages, optimizer_groups, seed_worker, atomic_checkpoint,
    rng_state, restore_rng, run_lock, runtime_versions, evaluate_validation, metric_report)
from research.aggressive.core import retry_registry_upsert
from research.plots import metric_figures, training_figures, comparison_figures

CFG=ROOT/'research/short_screening/mixup_v1.json'
RID='s40_convnext_tiny_mixup_strict_dev_seed42'
OUT=ROOT/'results/short_screening/mixup_v1'/RID
CK=ROOT/'checkpoints/short_screening/mixup_v1'/RID


def mixed_numerator(logits,a,b,w,lam):
    return lam*F.cross_entropy(logits.float(),a,weight=w,reduction='sum')+(1-lam)*F.cross_entropy(logits.float(),b,weight=w,reduction='sum')


def fingerprints():
    return {n:sha256(ROOT/n) for n in ['research/short_screening/controlled_mixup.py',
        'research/short_screening/mixup_v1.json','research/models.py','research/strict_train.py',
        'research/strict_protocol.py','research/common.py','research/plots.py','research/aggressive/core.py']}


def preflight():
    c=json.loads(CFG.read_text());train,val,w=development_data(c)
    torch.set_num_threads(4);torch.manual_seed(42)
    logits=torch.randn(32,7,requires_grad=True);a=torch.arange(32)%7;perm=torch.randperm(32);b=a[perm];lam=.37
    soft=lam*F.one_hot(a,7)+(1-lam)*F.one_hot(b,7)
    direct=-(soft*w*logits.log_softmax(1)).sum()/w[a].sum()
    full=mixed_numerator(logits,a,b,w,lam)/w[a].sum()
    micro=sum(mixed_numerator(z,aa,bb,w,lam) for z,aa,bb in zip(logits.split(16),a.split(16),b.split(16)))/w[a].sum()
    assert torch.allclose(direct,full,atol=1e-6) and torch.allclose(full,micro,atol=1e-6)
    assert torch.allclose(torch.autograd.grad(direct,logits,retain_graph=True)[0],torch.autograd.grad(micro,logits)[0],atol=1e-7)
    assert torch.allclose(mixed_numerator(logits,a,b,w,1)/w[a].sum(),F.cross_entropy(logits,a,weight=w))
    model=ResearchClassifier(c['model'],weights=c['weights'],dropout=c['head_dropout'])
    model.train();dataset=DevelopmentImages(train,c,True)
    x=torch.stack([dataset[i][0] for i in range(2)]);y=torch.tensor([dataset[i][1] for i in range(2)])
    z=model(.7*x+.3*x.flip(0));loss=mixed_numerator(z,y,y.flip(0),w,.7)/w[y].sum();loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(json.dumps(dict(cpu_preflight='passed',weighted_soft_label_and_microbatch_gradients_equivalent=True,
        model_forward_backward_finite=True,train_images=len(train),val_images=len(val),test_loaded=False,
        checkpoint_directory=relative(CK),result_directory=relative(OUT))))


def run(resume=False):
    c=json.loads(CFG.read_text());assert c['max_epochs']==20 and not c['test_evaluation']
    if (OUT/'summary.json').exists():print('Completed; preserved.');return
    OUT.mkdir(parents=True,exist_ok=resume);CK.mkdir(parents=True,exist_ok=resume)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    train,val,w=development_data(c);latest=CK/'latest.pt'
    cp=torch.load(latest,map_location='cpu',weights_only=False) if resume else None
    if cp:
        assert cp['config']==c and cp['fingerprints']==fingerprints() and cp['runtime_versions']==runtime_versions()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'train.log'),logging.StreamHandler()])
    model=ResearchClassifier(c['model'],weights=None if cp else c['weights'],dropout=c['head_dropout']).cuda()
    optimizer=torch.optim.AdamW(optimizer_groups(model,c),weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,threshold=0,threshold_mode='abs',min_lr=1e-7)
    scaler=torch.amp.GradScaler('cuda');w=w.cuda()
    mix_rng=np.random.RandomState(9042);history=[];best=f1best=None;updates=0;runtime=0
    meaningful=dict(accuracy=None,macro_f1=None,last_accuracy_epoch=0,last_f1_epoch=0)
    if cp:
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler']);scaler.load_state_dict(cp['scaler'])
        history=cp['history'];best=cp['best'];f1best=cp['f1best'];updates=cp['updates'];runtime=cp['runtime_seconds'];meaningful=cp['meaningful']
        restore_rng(cp['rng']);mix_rng.set_state(cp['mix_rng']);del cp
    from torchvision import models
    enum=models.get_model_weights(c['model'])[c['weights'].split('.')[-1]]
    cached=Path(torch.hub.get_dir())/'checkpoints'/enum.url.rsplit('/',1)[-1]
    write_json(OUT/'config.json',c)
    write_json(OUT/'verification.json',dict(split=partition_metadata()[1],fresh_imagenet_initialization=True,
        pretrained_file=relative(cached),pretrained_sha256=sha256(cached),class_order=CLASSES,
        fingerprints=fingerprints(),runtime_versions=runtime_versions(),test_loaded=False))
    trainset=DevelopmentImages(train,c,True);valset=DevelopmentImages(val,c)
    vloader=DataLoader(valset,batch_size=16,shuffle=False,num_workers=2,pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42))
    start=time.perf_counter()
    def payload():return dict(config=c,fingerprints=fingerprints(),runtime_versions=runtime_versions(),model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),history=history,best=best,f1best=f1best,rng=rng_state(True),mix_rng=mix_rng.get_state(),meaningful=meaningful,updates=updates,runtime_seconds=runtime+time.perf_counter()-start)
    def register(status,metrics=None):
        row=dict(experiment_id=RID,era='structured',record_kind='training_run',phase='post_test_development',protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],method=c['question'],model=c['model'],pretrained_weights=c['weights'],image_size=224,seed=42,augmentation=c['augmentation'],loss=c['loss'],epochs=len(history),config_path=relative(OUT/'config.json'),history_path=relative(OUT/'history.csv'),status=status,notes='Post-test development. Fresh ImageNet weights; Mixup-only ablation; hard20 cap; no test. Mixed-label training accuracy is not ordinary training accuracy.')
        if metrics:row.update({k:metrics[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']},best_epoch=best['epoch'],checkpoint=relative(CK/'best.pt'),checkpoint_sha256=sha256(CK/'best.pt'),metrics_path=relative(OUT/'validation_metrics.json'),plots_dir=relative(OUT/'figures'))
        retry_registry_upsert(row);write_json(OUT/'record.json',row)
    atomic_checkpoint(latest,payload());register('running')
    logging.info('START %s resume=%s epoch=%s; fresh ImageNet; hard20; checkpoint committed; test loader absent',RID,resume,len(history))
    write_json(OUT/'progress.json',dict(status='initialized',pid=os.getpid(),epoch=len(history),checkpoint_created=True))
    try:
        for epoch in range(len(history)+1,21):
            tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            loader=DataLoader(trainset,batch_size=16,shuffle=True,drop_last=True,num_workers=2,pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(42+epoch))
            assert len(loader)%2==0
            model.train();it=iter(loader);total_num=total_den=correct=0.;used=0;skipped=0
            for step in range(len(loader)//2):
                pairs=[next(it),next(it)];images=torch.cat([r[0] for r in pairs]).cuda();a=torch.cat([r[1] for r in pairs]).cuda()
                lam=float(mix_rng.beta(.2,.2));perm=torch.as_tensor(mix_rng.permutation(32),device='cuda');b=a[perm]
                mixed=lam*images+(1-lam)*images[perm];den=w[a].sum();optimizer.zero_grad(set_to_none=True)
                for offset in [0,16]:
                    with torch.autocast('cuda',dtype=torch.float16):z=model(mixed[offset:offset+16])
                    aa=a[offset:offset+16];bb=b[offset:offset+16];num=mixed_numerator(z,aa,bb,w,lam)
                    if not torch.isfinite(num):raise FloatingPointError('Nonfinite mixed training loss')
                    scaler.scale(num/den).backward();total_num+=float(num.detach());total_den+=float(w[aa].sum())
                    pred=z.argmax(1);correct+=float((lam*(pred==aa)+(1-lam)*(pred==bb)).sum());used+=len(aa)
                scaler.unscale_(optimizer)
                finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if finite:torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
                old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();accepted=scaler.get_scale()>=old_scale;updates+=int(accepted);skipped+=int(not accepted)
                if step==0 or (step+1)%50==0:
                    logging.info('epoch %s/20 step %s/%s mixed_loss=%.5f',epoch,step+1,len(loader)//2,total_num/total_den)
                    write_json(OUT/'progress.json',dict(status='training',pid=os.getpid(),epoch=epoch,step=step+1,checkpoint_created=True,first_update_confirmed=updates>0))
            ys,ps,ids,loss,precision=evaluate_validation(model,vloader,w,c,OUT,epoch)
            metrics=metric_report(np.asarray(ys),ps,loss)
            assert ids==val.image_id.tolist()
            preds=[dict(image_id=i,true_class=CLASSES[y],predicted_class=CLASSES[int(np.argmax(p))],**{f'p_{cl}':float(p[j]) for j,cl in enumerate(CLASSES)}) for i,y,p in zip(ids,ys,ps)]
            lrs=[g['lr'] for g in optimizer.param_groups];scheduler.step(metrics['accuracy'])
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            chosen=dict(epoch=epoch,metrics=metrics,predictions=preds,model=state)
            if best is None or metrics['accuracy']>best['metrics']['accuracy']:best=chosen;atomic_checkpoint(CK/'best.pt',dict(config=c,**chosen,fingerprints=fingerprints()))
            if f1best is None or metrics['macro_f1']>f1best['metrics']['macro_f1']:f1best=chosen;atomic_checkpoint(CK/'best_macro_f1.pt',dict(config=c,**chosen,fingerprints=fingerprints()))
            for key,delta in [('accuracy',.002),('macro_f1',.003)]:
                if meaningful[key] is None or metrics[key]-meaningful[key]>=delta-1e-12:meaningful[key]=metrics[key];meaningful['last_accuracy_epoch' if key=='accuracy' else 'last_f1_epoch']=epoch
            row=dict(epoch=epoch,train_loss=total_num/total_den,train_accuracy=correct/used,train_accuracy_definition='Mixup lambda-weighted label accuracy',val_loss=loss,val_accuracy=metrics['accuracy'],val_macro_precision=metrics['macro_precision'],val_macro_recall=metrics['macro_recall'],val_macro_f1=metrics['macro_f1'],backbone_lr=lrs[0],head_lr=lrs[1],backbone_lr_after=optimizer.param_groups[0]['lr'],head_lr_after=optimizer.param_groups[1]['lr'],amp_skipped_updates=skipped,epoch_seconds=time.perf_counter()-tick,peak_allocated_vram_mb=torch.cuda.max_memory_allocated()/1024**2,meaningful_accuracy_stale=epoch-meaningful['last_accuracy_epoch'],meaningful_f1_stale=epoch-meaningful['last_f1_epoch'])
            history.append(row);atomic_checkpoint(latest,payload());write_csv(OUT/'history.csv',history);write_csv(OUT/'lr_history.csv',[{k:r[k] for k in ['epoch','backbone_lr','head_lr','backbone_lr_after','head_lr_after']} for r in history])
            write_json(OUT/'validation_metrics_latest.json',metrics);write_csv(OUT/'validation_predictions_latest.csv',preds);register('running')
            logging.info('EPOCH %s accuracy=%.6f F1=%.6f; best=%s; hard cap20',epoch,metrics['accuracy'],metrics['macro_f1'],best['epoch'])
        for suffix,chosen in [('',best),('_macro_f1',f1best)]:
            write_json(OUT/f'validation_metrics{suffix}.json',chosen['metrics']);write_csv(OUT/f'validation_predictions{suffix}.csv',chosen['predictions'])
            write_csv(OUT/f'validation_probabilities{suffix}.csv',pd.DataFrame(chosen['predictions'])[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
            metric_figures(chosen['metrics'],OUT/f'figures{suffix}',f'S40 Mixup | strict development{suffix}')
        training_figures(pd.DataFrame(history),OUT/'figures','S40 Mixup training; mixed-label train accuracy',selection_metric='accuracy')
        p=pd.DataFrame(best['predictions']).set_index('image_id').loc[val.image_id][[f'p_{cl}' for cl in CLASSES]].to_numpy();fixed=[]
        for rid in ['s28_efficientnet_b0_final_strict_seed42','s30_efficientnet_v2_s_final_strict_seed42']:
            frame=pd.read_csv(ROOT/'results/structured_experiments'/rid/'validation_predictions.csv').set_index('image_id').loc[val.image_id]
            assert frame.true_class.tolist()==val.diagnosis.tolist();fixed.append(frame[[f'p_{cl}' for cl in CLASSES]].to_numpy())
        fused=(fixed[0]+p+fixed[1])/3;em=metric_report(val.label.to_numpy(),fused,float(-np.log(np.maximum(fused[np.arange(len(val)),val.label],1e-12)).mean()))
        write_json(OUT/'ensemble_metrics.json',em)
        ep=[dict(image_id=i,true_class=CLASSES[y],predicted_class=CLASSES[int(np.argmax(pr))],**{f'p_{cl}':float(pr[j]) for j,cl in enumerate(CLASSES)}) for i,y,pr in zip(val.image_id,val.label,fused)]
        write_csv(OUT/'ensemble_predictions.csv',ep)
        write_csv(OUT/'ensemble_probabilities.csv',pd.DataFrame(ep)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
        metric_figures(em,OUT/'ensemble_figures','S40 fixed equal ensemble; strict development')
        historical=pd.read_csv(ROOT/'results/structured_experiments/s29_convnext_tiny_final_strict_seed42/history.csv').iloc[:20];h=historical.loc[historical.val_accuracy.idxmax()]
        baseline=json.loads((ROOT/'results/final_strict/v1/ensemble/validation_metrics.json').read_text())
        parent=json.loads((ROOT/'results/structured_experiments/s29_convnext_tiny_final_strict_seed42/validation_metrics.json').read_text())
        rows=[dict(display_name='S29 control best through20',accuracy=h.val_accuracy,macro_f1=h.val_macro_f1),dict(display_name='S40 Mixup best through20',**{k:best['metrics'][k] for k in ['accuracy','macro_f1']}),dict(display_name='S29 full-run epoch33 reference',**{k:parent[k] for k in ['accuracy','macro_f1']}),dict(display_name='Frozen full strict ensemble',**{k:baseline[k] for k in ['accuracy','macro_f1']}),dict(display_name='S40 fixed ensemble',**{k:em[k] for k in ['accuracy','macro_f1']})]
        comparison_figures(rows,OUT/'comparison_figures','S40 controlled Mixup comparison')
        passed=em['accuracy']-baseline['accuracy']>=.005-1e-12 and em['macro_f1']>=baseline['macro_f1'] and em['per_class']['mel']['recall']>=baseline['per_class']['mel']['recall'] and em['per_class']['nv']['recall']>=baseline['per_class']['nv']['recall']-.01
        register('completed',best['metrics'])
        retry_registry_upsert(dict(experiment_id=RID+'_fixed_ensemble',era='structured',record_kind='inference_candidate',phase='post_test_development',protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],method='Fixed equal B0 + S40 ConvNeXt + V2S probability fusion',ensemble_members=['s28_efficientnet_b0_final_strict_seed42',RID,'s30_efficientnet_v2_s_final_strict_seed42'],ensemble_weights='[0.3333333333333333,0.3333333333333333,0.3333333333333333]',image_size=224,seed=42,epochs=0,metrics_path=relative(OUT/'ensemble_metrics.json'),plots_dir=relative(OUT/'ensemble_figures'),status='completed',decision='material_gate_passed' if passed else 'material_gate_failed',notes='Post-test development validation; no ensemble epoch search/test evaluation.',**{k:em[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
        write_json(OUT/'summary.json',dict(status='completed',epochs=20,best_epoch=best['epoch'],best_macro_f1_epoch=f1best['epoch'],best_accuracy=best['metrics']['accuracy'],best_macro_f1=f1best['metrics']['macro_f1'],ensemble_metrics=em,material_gate_passed=bool(passed),runtime_seconds=runtime+time.perf_counter()-start,stop_reason='hard20_epoch_cap',test_loaded=False,comparison=rows,decision='stop; no automatic next experiment'))
        write_json(OUT/'progress.json',dict(status='completed',pid=os.getpid(),epoch=20))
        logging.info('COMPLETE;20 epochs; material_gate=%s; STOP',passed)
        try:
            subprocess.run(['git','add',relative(OUT),'results/master_experiment_registry.csv'],cwd=ROOT,check=True)
            subprocess.run(['git','commit','-m','Close bounded S40 controlled Mixup experiment'],cwd=ROOT,check=True)
            subprocess.run(['git','push','origin','structured-research'],cwd=ROOT,check=True)
        except Exception as exc:write_json(OUT/'publication_pending.json',dict(error=repr(exc)))
    except Exception as exc:
        logging.exception('Stopped; last committed epoch and source artifacts preserved')
        write_json(OUT/'failure.json',dict(error=repr(exc),last_committed_epoch=len(history)));register('failed');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.check:preflight()
    elif args.run:
        with run_lock(RID):run(args.resume)
    else:parser.error('Use --check or --run [--resume]')
