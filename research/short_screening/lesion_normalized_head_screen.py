"""S74: one paired CPU screen of inverse-view-count training-loss weights."""
import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from research.common import ROOT,CLASSES,relative,sha256,write_csv,write_json
from research.plots import metric_figures,training_figures,save as save_figure
from research.registry import upsert
from research.short_screening.contrastive_head_screen import Head
from research.short_screening.kernel_head_screen import load,SPLIT,SPLIT_HASH
from research.short_screening.lesion_bag_screen import report

OUT=ROOT/'results/short_screening/s74_lesion_normalized_head_cpu'
CKPT=ROOT/'checkpoints/short_screening/s74_lesion_normalized_head_cpu'
CONTROL=ROOT/'results/short_screening/s73_matched_contrastive_head_cpu/ce_only'
CONTROL_CKPT=ROOT/'checkpoints/short_screening/s73_matched_contrastive_head_cpu/ce_only'
P_COLS=[f'p_{c}' for c in CLASSES]
PLAN=dict(
    question='Does removing extra training-loss influence from frequently photographed lesions improve generalization?',
    input='Existing square224 frozen ImageNet1k ConvNeXt-Tiny768dim features, only original development-training rows',
    matched_control='Exact S73 CE-only head, scaler, folds, seed, feature-dropout views, optimizer, LR schedule and epoch20 endpoint; control predictions independently reproduced from saved heads',
    change='Each training image receives additional multiplier1/n_training_views_of_its_lesion inside that fitting fold',
    class_weights='Unchanged normalized sqrt inverse-image-frequency CE weights fitted inside each training fold; not recomputed from lesion counts',
    loss='Sum per-image CE*class_weight*inverse_view_count divided by sum of those target weights per effective batch, both feature views included',
    caveat='Lesion influence equal before class multipliers in the whole fitting partition; batch-normalized SGD ratios do not guarantee exactly equal realized gradient influence',
    head='Feature dropout0.10; Linear768to256; GELU; dropout0.20; Linear256to128; Linear128to7',
    optimizer='AdamW lr0.001 weight_decay0.0001; cosine to0.00001 across20epochs; batch128; clip norm1',
    folds='Three fixed seed42 shuffled lesion-group folds; all7009 development-training images held out exactly once',
    gate='Accuracy improves >=0.01 absolute AND macro-F1 improves >=0.01; melanoma recall nondecreasing; positive net accuracy in >=2/3folds',
    conditional_validation='Only if gate passes: one paired full-training epoch20 control/candidate fit, followed by fixed existing exploratory validation; no new tuning',
    checkpoint_selection='Fixed final epoch20, no heldout epoch selection',
    search=False,gpu_used=False,test_loaded=False,cpu_threads=4,soft_wall_seconds=300,
    purpose='CPU diagnostic feature head, not CNN fine-tuning or an improved S53/test result; no automatic GPU launch')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'PREDECLARED_PLAN.json'
    if p.exists():assert json.loads(p.read_text())==PLAN
    else:write_json(p,PLAN)


def view_weights(groups):
    _,idx,counts=np.unique(np.asarray(groups),return_inverse=True,return_counts=True)
    w=1/counts[idx]
    np.testing.assert_allclose(np.bincount(idx,weights=w),np.ones(len(counts)),atol=1e-12)
    return w.astype(np.float32),counts


def weighted_loss(logits,y,class_weights,per_image):
    losses=F.cross_entropy(logits,y,reduction='none')
    weights=class_weights[y]*per_image
    assert (weights>0).all() and torch.isfinite(weights).all()
    return (losses*weights).sum()/weights.sum()


def check_loss():
    logits=torch.tensor([[2.,0.],[2.,0.],[0.,1.]],requires_grad=True)
    y=torch.tensor([0,0,1]);cw=torch.tensor([1.,2.])
    np.testing.assert_allclose(float(weighted_loss(logits,y,cw,torch.ones(3)).detach()),
                               float(F.cross_entropy(logits,y,weight=cw).detach()),rtol=1e-6)
    vw,_=view_weights(['a','a','b'])
    loss=weighted_loss(logits,y,cw,torch.tensor(vw))
    # Duplicate a's identical views have same class-weighted contribution as one.
    np.testing.assert_allclose(float(loss.detach()),float(F.cross_entropy(logits[[0,2]],y[[0,2]],weight=cw).detach()),rtol=1e-6)
    loss.backward();assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    write_json(OUT/'loss_preflight.json',dict(status='passed',uniform_matches_weighted_ce=True,
                  duplicate_identical_view_invariance=True,per_lesion_weight_sums_equal_one=True,finite_backward=True))


def fit(x,y,groups,fold,name,start):
    torch.manual_seed(42+fold);net=Head().cpu()
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=.00001)
    counts=np.bincount(y,minlength=7);assert (counts>0).all()
    cw=np.sqrt(len(y)/(7*counts));cw/=cw.mean()
    cw=torch.tensor(cw,dtype=torch.float32)
    vw,n_views=view_weights(groups)
    influence=np.ones(len(y),np.float32) if name=='image_weighted' else vw
    tx=torch.as_tensor(x,dtype=torch.float32);ty=torch.as_tensor(y,dtype=torch.long)
    iw=torch.tensor(influence,dtype=torch.float32);history=[]
    target=OUT/name/f'fold{fold}'
    write_csv(target/'training_weights.csv',[dict(training_row=i,lesion_id=str(g),class_name=CLASSES[int(label)],
                          inverse_view_count=float(vw[i]),applied_view_multiplier=float(influence[i]))
                          for i,(g,label) in enumerate(zip(groups,y))])
    for epoch in range(1,21):
        if time.perf_counter()-start>300:raise TimeoutError('Five-minute CPU limit; no GPU retry')
        net.train();order=torch.randperm(len(y),generator=torch.Generator().manual_seed(42000+fold*100+epoch))
        seen=correct=0;loss_sum=0.;lr=opt.param_groups[0]['lr']
        for ids in order.split(128):
            size=len(ids);x2=torch.cat([tx[ids],tx[ids]],0);y2=ty[ids].repeat(2)
            z,_=net(x2)
            loss=(F.cross_entropy(z,y2,weight=cw) if name=='image_weighted' else
                  weighted_loss(z,y2,cw,iw[ids].repeat(2)))
            assert torch.isfinite(loss) and z.device.type=='cpu'
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);opt.step()
            seen+=size;correct+=int(((z[:size]+z[size:]).argmax(1)==ty[ids]).sum())
            loss_sum+=float(loss.detach())*size
        sched.step();history.append(dict(epoch=epoch,train_loss=loss_sum/seen,train_accuracy=correct/seen,lr=lr))
        write_csv(target/'history.csv',history)
    path=CKPT/name/f'fold{fold}/final.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model_state=net.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=sched.state_dict(),
                    epoch=20,history=history,training_class_weights=cw,class_order=list(CLASSES),
                    kind='CPU feature head, not full CNN',seed=42+fold),path)
    training_figures(pd.DataFrame(history),target/'training_figures',f'S74 {name} fold{fold} | CPU head')
    return net,path,dict(fitting_images=len(y),fitting_lesions=len(n_views),multiimage_lesions=int((n_views>1).sum()),
                         maximum_views=int(n_views.max()),class_weights=cw.tolist())


def predict(net,x):
    net.eval()
    with torch.inference_mode():
        return torch.cat([F.softmax(net(t)[0],dim=1) for t in torch.from_numpy(x).split(512)]).numpy()


def save_scores(frame,y,p,path,stem):
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
    m=report(y,p);write_json(path/(stem+'_metrics.json'),m)
    rows=frame.copy();rows['true_class']=[CLASSES[i] for i in y];rows['predicted_class']=[CLASSES[i] for i in p.argmax(1)];rows[P_COLS]=p
    write_csv(path/(stem+'_predictions.csv'),rows.to_dict('records'))
    write_csv(path/(stem+'_probabilities.csv'),rows[['image_id']+P_COLS].to_dict('records'))
    title='train-only lesion-group OOF' if stem=='training_meta_oof' else 'exploratory validation diagnostic'
    metric_figures(m,path/'figures',f'S74 {path.name} | {title}')
    check=pd.read_csv(path/(stem+'_predictions.csv'));assert check.image_id.tolist()==frame.image_id.tolist()
    rem=report(check.true_class.map(dict(zip(CLASSES,range(7)))).to_numpy(),check[P_COLS].to_numpy())
    assert rem['confusion_matrix']==m['confusion_matrix'] and abs(rem['macro_f1']-m['macro_f1'])<1e-12
    if stem=='training_meta_oof':
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,3));ax.bar(CLASSES,np.asarray(m['confusion_matrix']).sum(1))
        ax.set(ylabel='Train-fold heldout images',title=f'S74 {path.name} | training OOF support')
        save_figure(fig,path/'figures','class_support')
    return m


def compare(metrics,path,title,xlabel):
    import matplotlib.pyplot as plt
    names=list(metrics);pos=np.arange(len(names));fig,ax=plt.subplots(figsize=(9,3.5))
    for offset,field,label in [(-.18,'accuracy','Accuracy'),(.18,'macro_f1','Macro-F1')]:
        bars=ax.barh(pos+offset,[metrics[n][field] for n in names],.35,label=label)
        ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
    ax.set(yticks=pos,yticklabels=names,xlim=(0,1),title=title,xlabel=xlabel)
    ax.invert_yaxis();ax.legend(loc='upper left',bbox_to_anchor=(1.02,1));ax.grid(axis='x',alpha=.2)
    save_figure(fig,path,'model_comparison')


def main():
    prepare()
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    lock=OUT/'run.lock';handle=lock.open('x')
    try:
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',encoding='utf-8'),logging.StreamHandler()])
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);check_loss();start=time.perf_counter()
        logging.info('START fixed inverse-lesion-view CE weighting; CPU only, no validation/test images')
        assert sha256(SPLIT)==SPLIT_HASH
        ids=pd.read_csv(SPLIT,usecols=['image_id','lesion_id','split']);train_ids=ids.loc[ids.split=='train'].copy()
        locked=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','lesion_id','split'])
        test=locked.loc[locked.split=='test']
        assert not set(train_ids.image_id)&set(test.image_id) and not set(train_ids.lesion_id)&set(test.lesion_id)
        frame,x,y,source=load('train',train_ids);assert len(frame)==7009
        groups=frame.lesion_id.to_numpy()
        ref_path=CONTROL/'training_meta_oof_predictions.csv';ref_hash=sha256(ref_path)
        ref=pd.read_csv(ref_path);assert ref.image_id.tolist()==frame.image_id.tolist()
        assert ref.true_class.tolist()==[CLASSES[i] for i in y]
        base_p=ref[P_COLS].to_numpy();p=np.full((len(y),7),np.nan)
        assignment=np.full(len(y),-1,int);coverage=np.zeros(len(y),int);rows=[];saved_heads={};control_hashes={}
        for fold,(f,h) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(x,y,groups),1):
            assert not set(groups[f])&set(groups[h]);coverage[h]+=1;assignment[h]=fold
            assert np.all(ref.meta_fold.iloc[h].to_numpy()==fold)
            scaler=StandardScaler().fit(x[f]);fx=scaler.transform(x[f]).astype(np.float32);hx=scaler.transform(x[h]).astype(np.float32)
            original=CONTROL_CKPT/f'fold{fold}/final.pt';control_hashes[relative(original)]=sha256(original)
            state=torch.load(original,map_location='cpu',weights_only=True);assert state['epoch']==20
            control=Head().cpu();control.load_state_dict(state['model_state'])
            np.testing.assert_allclose(predict(control,hx),base_p[h],rtol=1e-6,atol=1e-7)
            net,path,weight_info=fit(fx,y[f],groups[f],fold,'lesion_normalized',start)
            p[h]=predict(net,hx);saved_heads[relative(path)]=sha256(path)
            scale_path=CKPT/f'fold{fold}_standard_scaler.npz'
            np.savez_compressed(scale_path,mean=scaler.mean_,scale=scaler.scale_)
            record=dict(fold=fold,**weight_info,control_accuracy=report(y[h],base_p[h])['accuracy'],
                        candidate_accuracy=report(y[h],p[h])['accuracy'],
                        net_correct=int((p[h].argmax(1)==y[h]).sum()-(base_p[h].argmax(1)==y[h]).sum()))
            rows.append(record);write_json(OUT/f'fold{fold}_weight_manifest.json',record)
            logging.info('Fold %d control %.4f%% candidate %.4f%% net %+d',fold,record['control_accuracy']*100,record['candidate_accuracy']*100,record['net_correct'])
        assert (coverage==1).all();frame['meta_fold']=assignment
        write_csv(OUT/'fold_assignments.csv',frame[['image_id','lesion_id','meta_fold']].to_dict('records'))
        write_csv(OUT/'fold_comparison.csv',[{k:v for k,v in r.items() if k!='class_weights'} for r in rows])
        metrics={'image_weighted':save_scores(frame,y,base_p,OUT/'image_weighted','training_meta_oof'),
                 'lesion_normalized':save_scores(frame,y,p,OUT/'lesion_normalized','training_meta_oof')}
        a,b=metrics.values();positive=sum(r['net_correct']>0 for r in rows)
        passed=b['accuracy']>=a['accuracy']+.01-1e-12 and b['macro_f1']>=a['macro_f1']+.01-1e-12 and b['per_class']['mel']['recall']>=a['per_class']['mel']['recall']-1e-12 and positive>=2
        compare(metrics,OUT/'comparison_figures','S74 inverse-view-count loss influence | CPU feature heads','Train-only lesion-group out-of-fold score')
        old=base_p.argmax(1)==y;new=p.argmax(1)==y
        summary=dict(status='completed',metrics=metrics,positive_folds=positive,gate_passed=bool(passed),
                     gained=int((~old&new).sum()),lost=int((old&~new).sum()),net_correct=int(new.sum()-old.sum()),
                     validation_scored=False,test_loaded=False,gpu_used=False,epochs_per_fold_head=20,
                     decision='Gate passed; fixed validation followup, no automatic GPU launch' if passed else 'Reject inverse-view-count recipe; no validation, parameter rescue or GPU run')
        sources=[source]
        if passed:
            val_ids=ids.loc[ids.split=='val'].copy()
            assert not set(val_ids.image_id)&set(test.image_id) and not set(val_ids.lesion_id)&set(test.lesion_id)
            vf,vx,vy,vs=load('val',val_ids);sources.append(vs)
            scaler=StandardScaler().fit(x);fx=scaler.transform(x).astype(np.float32);vx=scaler.transform(vx).astype(np.float32)
            vm={}
            for name in ['image_weighted','lesion_normalized']:
                net,path,_=fit(fx,y,groups,0,name,start);saved_heads[relative(path)]=sha256(path)
                vm[name]=save_scores(vf,vy,predict(net,vx),OUT/name/'development_validation','validation')
            compare(vm,OUT/'development_comparison_figures','S74 paired full-fit frozen-feature diagnostic','Exploratory validation score')
            summary.update(validation_scored=True,validation_metrics=vm)
        assert sha256(ref_path)==ref_hash and all(sha256(ROOT/s['path'])==s['sha256'] for s in sources)
        assert all(sha256(ROOT/p)==h for p,h in control_hashes.items())
        write_json(OUT/'source_manifest.json',dict(feature_files=sources,control_prediction_path=relative(ref_path),
                   control_prediction_sha256=ref_hash,control_checkpoints=control_hashes,candidate_checkpoints=saved_heads,
                   script_sha256=sha256(ROOT/'research/short_screening/lesion_normalized_head_screen.py'),split_sha256=SPLIT_HASH))
        write_json(OUT/'verification.json',dict(status='passed',loss_preflight_passed=True,control_predictions_reproduced_from_saved_heads=True,
                    training_fold_view_counts_and_class_weights_only=True,scaling_fit_only_inside_training_fold=True,
                    fold_lesion_overlap=0,OOF_coverage=int(coverage.sum()),sources_and_control_checkpoints_unchanged=True,
                    saved_metrics_and_confusion_recomputed=True,validation_feature_cache_loaded=bool(passed),test_labels_read=False,gpu_used=False))
        m=b
        upsert(dict(experiment_id='s74_lesion_normalized_ce_head_train_group_cv_seed42',era='structured',record_kind='cpu_feature_head_training_cv',
                    phase='post_test_development_preflight',protocol='training_only_lesion_group_3fold',evaluation_split='training_meta_oof',
                    split_manifest=relative(SPLIT),split_sha256=SPLIT_HASH,model='frozen_convnext_tiny_imagenet1k_plus_mlp_head',
                    method=PLAN['change'],epochs=20,seed=42,status='completed',decision='gate_passed' if passed else 'gate_failed',
                    metrics_path=relative(OUT/'lesion_normalized/training_meta_oof_metrics.json'),plots_dir=relative(OUT/'lesion_normalized/figures'),
                    checkpoint=relative(CKPT/'lesion_normalized/fold1/final.pt'),
                    notes='Three CPU feature heads, fixed epoch20; same verified S73 CE control. Not full CNN, S53 ensemble or test result. No GPU.',
                    **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
        summary['runtime_seconds']=time.perf_counter()-start;write_json(OUT/'summary.json',summary)
        logging.info('COMPLETE gate=%s runtime=%.1fs',passed,summary['runtime_seconds'])
        print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
    finally:handle.close();lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else main()
