"""S73: matched CPU feature-head CE versus CE+SupCon preflight.

Frozen ImageNet features only. Stochastic feature dropout is a cheap proxy,
not image augmentation or end-to-end supervised contrastive CNN training.
"""
import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from research.common import ROOT,CLASSES,relative,sha256,write_csv,write_json
from research.plots import metric_figures,comparison_figures,training_figures
from research.registry import upsert
from research.short_screening.kernel_head_screen import load,SPLIT,SPLIT_HASH,WEIGHT_HASH
from research.short_screening.lesion_bag_screen import report

OUT=ROOT/'results/short_screening/s73_matched_contrastive_head_cpu'
CKPT=ROOT/'checkpoints/short_screening/s73_matched_contrastive_head_cpu'
P_COLS=[f'p_{c}' for c in CLASSES]
PLAN=dict(
    question='Does a supervised contrastive feature regularizer improve group-heldout class separation beyond the identical cross-entropy head?',
    source='https://arxiv.org/abs/2004.11362',
    paper_scope='Supervised contrastive learning clusters same-class embeddings; this is an adapted frozen-feature joint-loss preflight, not replication of its image-augmented two-stage procedure',
    inputs='Cached ImageNet1k ConvNeXt-Tiny FP32 square224 features,768 dimensions; no fine-tuned CNN features or validation-selected ensemble probabilities',
    folds='3 fixed shuffled StratifiedGroupKFold lesion_id folds seed42; original7009 development-training images only',
    matched_head='Train-fold StandardScaler; feature dropout0.10 -> Linear768to256 -> GELU -> dropout0.20 -> Linear256to128 -> Linear128to7 classifier',
    views='Two independent dropout feature views for BOTH control and candidate; not distinct image augmentation or real companion-image inference',
    optimizer='AdamW lr0.001 weight_decay0.0001; cosine LR to0.00001 across20 epochs; batch128; gradient clip norm1',
    loss_control='Training-fold sqrt inverse-frequency weighted CE over both views',
    loss_candidate='Same CE +0.1*SupCon; temperature0.1; L2 normalized128dim embeddings; average over all same-class nonself positives',
    imbalance='Same fold-training-only mean1 sqrt inverse-frequency CE weights; contrastive anchors unweighted',
    checkpoint_selection='Epoch20 fixed final; no heldout epoch selection, no parameter/loss-weight search',
    gate='Pooled candidate accuracy >= control+0.01; macro-F1 >= control+0.01; melanoma recall nondecreasing; positive accuracy net in at least2/3 folds',
    continuation='Passing permits proposing a matched bounded image-space pilot; no automatic GPU launch and no claim about S53/test improvement',
    validation_scored=False,test_loaded=False,gpu_used=False,cpu_threads=4,
    hard_epochs_per_head=20,soft_wall_budget_seconds=300,
)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'PREDECLARED_PLAN.json'
    if path.exists():assert json.loads(path.read_text())==PLAN
    else:write_json(path,PLAN)


def supcon(h,y,temperature=.1):
    """Mean nonself same-class log-probability; FP32 logsumexp, no label leakage."""
    z=F.normalize(h.float(),dim=1)
    logits=z@z.T/temperature
    own=torch.eye(len(y),device=y.device,dtype=torch.bool)
    positive=y[:,None].eq(y[None,:]) & ~own
    count=positive.sum(1)
    if not (count>0).all():raise ValueError('Every anchor requires a positive view')
    logits=logits.masked_fill(own,float('-inf'))
    logp=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    # Multiplying masked -inf by zero would produce NaN; use selection.
    selected=torch.where(positive,logp,torch.zeros_like(logp))
    return -(selected.sum(1)/count).mean()


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Sequential(nn.Dropout(.1),nn.Linear(768,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,128))
        self.classifier=nn.Linear(128,7)
    def forward(self,x):
        h=self.embedding(x);return self.classifier(h),h


def check_loss():
    generator=torch.Generator().manual_seed(7)
    h=torch.randn(8,128,generator=generator,requires_grad=True)
    labels=torch.tensor([0,1,2,3,0,1,2,3])
    loss=supcon(h,labels);assert torch.isfinite(loss)
    # Independent scalar reference catches denominator/self/positive errors.
    z=F.normalize(h.detach().double(),dim=1);values=[]
    for i in range(8):
        js=[j for j in range(8) if j!=i]
        positive=[j for j in js if labels[j]==labels[i]]
        scores=torch.stack([(z[i]*z[j]).sum()/.1 for j in js])
        denom=torch.logsumexp(scores,dim=0)
        values.append(torch.stack([denom-(z[i]*z[j]).sum()/.1 for j in positive]).mean())
    np.testing.assert_allclose(float(loss.detach()),float(torch.stack(values).mean()),rtol=1e-6)
    q=torch.tensor([7,0,5,2,4,1,3,6])
    np.testing.assert_allclose(float(supcon(h[q],labels[q]).detach()),float(loss.detach()),rtol=1e-6)
    loss.backward();assert torch.isfinite(h.grad).all() and h.grad.abs().sum()>0
    write_json(OUT/'loss_preflight.json',dict(status='passed',scalar_reference_agrees=True,
        self_excluded=True,permutation_invariant=True,finite_backward=True,gpu_used=False))


def train_head(x,y,fold,name,start):
    torch.manual_seed(42+fold)
    net=Head().cpu()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=.00001)
    counts=np.bincount(y,minlength=7);assert (counts>0).all()
    w=np.sqrt(len(y)/(7*counts));w/=w.mean()
    weights=torch.tensor(w,dtype=torch.float32)
    tx=torch.as_tensor(x,dtype=torch.float32);ty=torch.as_tensor(y,dtype=torch.long)
    history=[]
    target=OUT/name/f'fold{fold}'
    for epoch in range(1,21):
        if time.perf_counter()-start>300:raise TimeoutError('Five-minute CPU budget; preserve heads/history without GPU retry')
        net.train();order=torch.randperm(len(y),generator=torch.Generator().manual_seed(42000+fold*100+epoch))
        loss_sum=ce_sum=contrast_sum=0.;correct=seen=0;lr=optimizer.param_groups[0]['lr']
        for ids in order.split(128):
            x2=torch.cat([tx[ids],tx[ids]],dim=0);y2=ty[ids].repeat(2)
            logits,h=net(x2);ce=F.cross_entropy(logits,y2,weight=weights)
            contrast=supcon(h,y2) if name=='ce_supcon' else h.new_zeros(())
            loss=ce+.1*contrast
            assert torch.isfinite(loss) and logits.device.type=='cpu'
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);optimizer.step()
            size=len(ids);seen+=size;correct+=int(((logits[:size]+logits[size:]).argmax(1)==ty[ids]).sum())
            loss_sum+=float(loss.detach())*size;ce_sum+=float(ce.detach())*size;contrast_sum+=float(contrast.detach())*size
        scheduler.step()
        row=dict(epoch=epoch,train_loss=loss_sum/seen,train_ce_loss=ce_sum/seen,
                 train_contrastive_loss=contrast_sum/seen,train_accuracy=correct/seen,lr=lr)
        history.append(row)
        write_csv(target/'history.csv',history)
    path=CKPT/name/f'fold{fold}'/'final.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model_state=net.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
                    epoch=20,history=history,training_class_weights=weights,class_order=list(CLASSES),
                    kind='CPU feature head; not CNN checkpoint',seed=42+fold),path)
    training_figures(pd.DataFrame(history),target/'training_figures',f'S73 {name} fold{fold} | CPU head training')
    return net,path


def save_predictions(frame,y,p,path):
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
    m=report(y,p);write_json(path/'training_meta_oof_metrics.json',m)
    rows=frame.copy();rows['true_class']=[CLASSES[i] for i in y];rows['predicted_class']=[CLASSES[i] for i in p.argmax(1)];rows[P_COLS]=p
    write_csv(path/'training_meta_oof_predictions.csv',rows.to_dict('records'))
    write_csv(path/'training_meta_oof_probabilities.csv',rows[['image_id']+P_COLS].to_dict('records'))
    metric_figures(m,path/'figures',f'S73 {path.name} | training-only lesion-group OOF')
    reread=pd.read_csv(path/'training_meta_oof_predictions.csv')
    assert reread.image_id.tolist()==frame.image_id.tolist()
    rem=report(reread.true_class.map(dict(zip(CLASSES,range(7)))).to_numpy(),reread[P_COLS].to_numpy())
    assert rem['confusion_matrix']==m['confusion_matrix'] and abs(rem['macro_f1']-m['macro_f1'])<1e-12
    return m


def main():
    prepare()
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    lock=OUT/'run.lock';handle=lock.open('x')
    try:
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',encoding='utf-8'),logging.StreamHandler()])
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        check_loss();start=time.perf_counter()
        logging.info('START fixed CE vs CE+SupCon CPU head comparison; no validation/test/GPU')
        assert sha256(SPLIT)==SPLIT_HASH
        identities=pd.read_csv(SPLIT,usecols=['image_id','lesion_id','split'])
        train_ids=identities.loc[identities.split=='train'].copy()
        strict=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','lesion_id','split'])
        test=strict.loc[strict.split=='test']
        assert not set(train_ids.image_id)&set(test.image_id) and not set(train_ids.lesion_id)&set(test.lesion_id)
        frame,x,y,source=load('train',train_ids);assert len(frame)==7009
        groups=frame.lesion_id.to_numpy();names=['ce_only','ce_supcon']
        ps={name:np.full((len(y),7),np.nan) for name in names}
        assignments=np.full(len(y),-1,int);coverage=np.zeros(len(y),int);records=[];checkpoints={}
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42)
        for fold,(fit,held) in enumerate(splitter.split(x,y,groups),1):
            assert not set(groups[fit])&set(groups[held]);coverage[held]+=1;assignments[held]=fold
            scaler=StandardScaler().fit(x[fit])
            fx=scaler.transform(x[fit]).astype(np.float32);hx=scaler.transform(x[held]).astype(np.float32)
            cache=CKPT/f'fold{fold}_standard_scaler.npz';cache.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(cache,mean=scaler.mean_,scale=scaler.scale_)
            record=dict(fold=fold,fit_images=len(fit),heldout_images=len(held),lesion_overlap=0)
            for name in names:
                net,path=train_head(fx,y[fit],fold,name,start);net.eval()
                with torch.inference_mode():
                    p=torch.cat([F.softmax(net(part)[0],dim=1) for part in torch.from_numpy(hx).split(512)]).numpy()
                ps[name][held]=p;m=report(y[held],p)
                record[name+'_accuracy']=m['accuracy'];record[name+'_macro_f1']=m['macro_f1']
                checkpoints[relative(path)]=sha256(path)
            record['net_correct']=int((ps[names[1]][held].argmax(1)==y[held]).sum()-(ps[names[0]][held].argmax(1)==y[held]).sum())
            records.append(record);write_csv(OUT/'fold_comparison.csv',records)
            logging.info('Fold %d CE %.4f%% SupCon %.4f%% net %+d',fold,record['ce_only_accuracy']*100,record['ce_supcon_accuracy']*100,record['net_correct'])
        assert (coverage==1).all()
        frame['meta_fold']=assignments;write_csv(OUT/'fold_assignments.csv',frame[['image_id','lesion_id','meta_fold']].to_dict('records'))
        metrics={name:save_predictions(frame,y,p,OUT/name) for name,p in ps.items()}
        a,b=metrics[names[0]],metrics[names[1]];positive=sum(r['net_correct']>0 for r in records)
        passed=b['accuracy']>=a['accuracy']+.01-1e-12 and b['macro_f1']>=a['macro_f1']+.01-1e-12 and b['per_class']['mel']['recall']>=a['per_class']['mel']['recall']-1e-12 and positive>=2
        old=ps[names[0]].argmax(1)==y;new=ps[names[1]].argmax(1)==y
        comparison_figures([dict(display_name=name,accuracy=m['accuracy'],macro_f1=m['macro_f1']) for name,m in metrics.items()],OUT/'comparison_figures','S73 matched CPU feature-head contrastive preflight')
        for name,m in metrics.items():
            upsert(dict(experiment_id=f's73_{name}_train_group_cv_seed42',era='structured',record_kind='cpu_feature_head_training_cv',
                       phase='post_test_development_preflight',protocol='training_only_lesion_group_3fold',evaluation_split='training_meta_oof',
                       split_manifest=relative(SPLIT),split_sha256=SPLIT_HASH,model='frozen_convnext_tiny_imagenet1k_plus_mlp_head',
                       method=PLAN['loss_control'] if name=='ce_only' else PLAN['loss_candidate'],epochs=20,seed=42,status='completed',
                       decision='gate_passed' if passed else 'gate_failed',metrics_path=relative(OUT/name/'training_meta_oof_metrics.json'),
                       plots_dir=relative(OUT/name/'figures'),checkpoint=relative(CKPT/name/'fold1/final.pt'),
                       notes='Three CPU fold heads; fixed epoch20, no heldout epoch selection. Feature dropout proxy, not full CNN image SupCon. No validation/test/GPU.',
                       **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
        assert sha256(ROOT/source['path'])==source['sha256']
        write_json(OUT/'source_manifest.json',dict(feature_source=source,class_order=list(CLASSES),split_sha256=SPLIT_HASH,checkpoints=checkpoints))
        write_json(OUT/'verification.json',dict(status='passed',loss_reference_and_backward_passed=True,paired_seeds_architecture_augmentation=True,
            scalers_and_class_weights_training_fold_only=True,heldout_coverage=int(coverage.sum()),fold_lesion_overlap=0,
            source_feature_cache_unchanged=True,metrics_and_confusion_recomputed=True,validation_feature_cache_loaded=False,
            original_test_labels_read=False,gpu_used=False))
        summary=dict(status='completed',metrics=metrics,gate_passed=bool(passed),positive_folds=positive,
            gained=int((~old&new).sum()),lost=int((old&~new).sum()),net_correct=int(new.sum()-old.sum()),
            runtime_seconds=time.perf_counter()-start,epochs_per_fold_head=20,folds=3,gpu_used=False,test_loaded=False,
            validation_scored=False,decision='May propose matched image-space pilot; no automatic launch' if passed else 'Reject this fixed feature-space SupCon preflight; no GPU launch or loss-weight rescue search')
        write_json(OUT/'summary.json',summary);logging.info('COMPLETE gate=%s runtime=%.1fs',passed,summary['runtime_seconds']);print(json.dumps(summary,indent=2))
    finally:
        handle.close();lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else main()
