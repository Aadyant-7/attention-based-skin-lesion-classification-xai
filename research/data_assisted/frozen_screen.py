"""One fixed CPU data-expansion comparison plus train-only group confirmation."""
import json,logging,time
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv
from research.models import ResearchClassifier
from research.strict_train import metric_report,run_lock
from research.short_screening.convnextv2_transfer import config,development
from research.aggressive.core import predictions,retry_registry_upsert
from research.plots import metric_figures,comparison_figures

OUT=ROOT/'results/data_assisted/s66_s67_frozen_data_screen'
CACHE=ROOT/'.cache/s66_s67_frozen_data_screen'
CURATED=ROOT/'results/data_assisted/clearance_v1'
HAM_CACHE=ROOT/'.cache/s64_paired_preprocessing_cpu'
WEIGHTS=ROOT/'.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth'
NAMES=['ham_only','ham_plus_external']


class ExternalImages(Dataset):
    def __init__(self,frame,c):
        self.frame=frame
        self.transform=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
            transforms.ToTensor(),transforms.Normalize(c['normalization_mean'],c['normalization_std'])])
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i];path=ROOT/row.path
        assert sha256(path)==row.sha256
        with Image.open(path) as image:x=self.transform(image.convert('RGB'))
        return x,int(row.label),row.image_id


def features(frame,c,weight_hash,manifest_hash,start):
    path=CACHE/'external_square224.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            assert f['ids'].tolist()==frame.image_id.tolist() and f['y'].tolist()==frame.label.tolist()
            assert str(f['weights_sha256'])==weight_hash and str(f['manifest_sha256'])==manifest_hash
            x=f['features'].copy()
    else:
        torch.set_num_threads(4);torch.manual_seed(42)
        net=ResearchClassifier('convnext_tiny',weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1').cpu().eval()
        for p in net.parameters():p.requires_grad_(False)
        loader=DataLoader(ExternalImages(frame,c),batch_size=8,shuffle=False,num_workers=0)
        vectors=[];ids=[];ys=[]
        with torch.inference_mode():
            for step,(images,y,names) in enumerate(loader):
                assert images.device.type=='cpu' and torch.isfinite(images).all()
                z=net.head[:3](net.features(images));assert z.device.type=='cpu' and torch.isfinite(z).all()
                vectors.append(z.numpy());ids.extend(names);ys.extend(y.tolist())
                if step%50==0:logging.info('External frozen CPU features %d/%d',len(ids),len(frame))
                if time.perf_counter()-start>1200:raise TimeoutError('20-minute CPU feature budget exceeded; no automatic GPU retry')
        x=np.concatenate(vectors)
        assert ids==frame.image_id.tolist() and ys==frame.label.tolist()
        np.savez_compressed(path,features=x,ids=np.array(ids),y=np.array(ys),weights_sha256=np.array(weight_hash),manifest_sha256=np.array(manifest_hash))
    assert x.shape==(len(frame),768) and np.isfinite(x).all()
    return x


def weights(y,fp32=False):
    counts=np.bincount(y,minlength=7);assert (counts>0).all()
    w=np.sqrt(len(y)/(7*counts));w/=w.mean()
    if fp32:w=w.astype(np.float32)
    return w[y]


def fit(x,y,name,fp32=False):
    head=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
    with threadpool_limits(limits=4):
        head.fit(x,y,logisticregression__sample_weight=weights(y,fp32))
    assert head[-1].classes_.tolist()==list(range(7)) and head[-1].n_iter_.max()<1000
    joblib.dump(head,CACHE/(name+'.joblib'));return head


def evaluate(frame,p,target,title,stem):
    y=frame.label.to_numpy()
    assert p.shape==(len(frame),7) and np.isfinite(p).all() and np.allclose(p.sum(1),1)
    m=metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))
    write_json(target/(stem+'_metrics.json'),m)
    rows=predictions(frame.image_id,y,p)
    write_csv(target/(stem+'_predictions.csv'),rows)
    write_csv(target/(stem+'_probabilities.csv'),pd.DataFrame(rows)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
    metric_figures(m,target/'figures',title)
    saved=pd.read_csv(target/(stem+'_predictions.csv'))
    assert saved.image_id.tolist()==frame.image_id.tolist()
    check=metric_report(y,saved[[f'p_{cl}' for cl in CLASSES]].to_numpy(),m['loss'])
    assert check['confusion_matrix']==m['confusion_matrix'] and abs(check['macro_f1']-m['macro_f1'])<1e-12
    return m


def recall(m,cl='mel'):
    cm=np.array(m['confusion_matrix']);i=list(CLASSES).index(cl);return float(cm[i,i]/cm[i].sum())


def paired(frame,pa,pb,ma,mb,target):
    y=frame.label.to_numpy();a=pa.argmax(1)==y;b=pb.argmax(1)==y
    rows=[]
    for i,cl in enumerate(CLASSES):
        idx=y==i
        rows.append(dict(class_name=cl,support=int(idx.sum()),control_recall=recall(ma,cl),candidate_recall=recall(mb,cl),
            gained=int((~a&b&idx).sum()),lost=int((a&~b&idx).sum()),net_correct_change=int(b[idx].sum()-a[idx].sum())))
    write_csv(target/'class_comparison.csv',rows)
    write_csv(target/'paired_changes.csv',[dict(image_id=frame.iloc[i].image_id,true_class=CLASSES[y[i]],
        control_class=CLASSES[pa[i].argmax()],candidate_class=CLASSES[pb[i].argmax()],
        change='gained' if b[i] else 'lost') for i in np.flatnonzero(a!=b)])
    return dict(gained=int((~a&b).sum()),lost=int((a&~b).sum()),net_correct_change=int(b.sum()-a.sum()),
        accuracy_delta=float(mb['accuracy']-ma['accuracy']),macro_f1_delta=float(mb['macro_f1']-ma['macro_f1']),
        melanoma_recall_control=recall(ma),melanoma_recall_candidate=recall(mb))


def main():
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
    start=time.perf_counter();c=config();train,val,_=development(c)
    cleared=json.loads((CURATED/'pixel_clearance_summary.json').read_text())
    manifest=CURATED/'pixel_screened_candidate_manifest.csv';manifest_hash=sha256(manifest)
    assert manifest_hash==cleared['candidate_manifest_sha256'] and cleared['ready_for_bounded_development_screen']
    ext=pd.read_csv(manifest);ext['label']=ext.class_name.map(dict(zip(CLASSES,range(7))))
    assert ext.label.notna().all() and ext.image_id.is_unique and ext.lesion_id.is_unique
    assert not set(ext.image_id)&set(pd.concat([train,val]).image_id)
    assert not set(ext.lesion_id)&set(pd.concat([train,val]).lesion_id)
    assert len(ext)==3323
    extfit=np.flatnonzero(ext.external_partition=='external_train');extheld=np.flatnonzero(ext.external_partition=='external_preflight_val')
    assert len(extfit)==2823 and len(extheld)==500
    plan=dict(experiments=['S66 paired exploratory feature-head screen','S67 train-only3fold lesion-group consistency'],
        question='Does curated external dermoscopy training data improve matched frozen-feature classifiers?',
        weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1',precision='CPU FP32',transform='Square224 bilinear; ImageNet normalization; identity only',
        head='Fold-local StandardScaler + LogisticRegression C1 max_iter1000 seed42; normalized sqrt inverse-frequency sample weights',
        candidate='Same classifier plus fixed2823 external training lesions; same external set in each HAM group-CV training fold',
        external_preflight='500 fixed external development-validation lesions; descriptive domain shift only; never fit',
        validation_gate='accuracy+0.01 AND macroF1+0.01 AND melanoma recall nondecreasing',
        group_gate='pooled accuracy+0.01 AND macroF1 nondecreasing AND melanoma recall nondecreasing AND2/3 folds positive',
        limit='No source/subset/head/weight search; no GPU or original test; no automatic training regardless of score',
        external_manifest_sha256=manifest_hash,independence_limitations=cleared['not_a_certificate_of_final_test_independence'])
    if (OUT/'PREDECLARED_PLAN.json').exists():assert json.loads((OUT/'PREDECLARED_PLAN.json').read_text())==plan
    else:write_json(OUT/'PREDECLARED_PLAN.json',plan)
    old=json.loads((ROOT/'results/short_screening/s64_paired_preprocessing_cpu/source_manifest.json').read_text())
    assert WEIGHTS.is_file();wh=sha256(WEIGHTS);assert wh==old['weights_sha256']
    xs={};feature_hashes={}
    for part,frame in [('train',train),('val',val)]:
        path=HAM_CACHE/f'square224_{part}.npz'
        with np.load(path,allow_pickle=False) as f:
            assert f['ids'].tolist()==frame.image_id.tolist() and f['y'].tolist()==frame.label.tolist()
            assert str(f['weights_sha256'])==wh;xs[part]=f['features'].copy()
        assert xs[part].shape==(len(frame),768) and np.isfinite(xs[part]).all()
        feature_hashes[relative(path)]=sha256(path)
    ex=features(ext,c,wh,manifest_hash,start);ey=ext.label.to_numpy()
    feature_hashes[relative(CACHE/'external_square224.npz')]=sha256(CACHE/'external_square224.npz')
    source=dict(weights_sha256=wh,weights_path=relative(WEIGHTS),external_manifest_sha256=manifest_hash,
        split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],feature_hashes=feature_hashes,
        script_sha256=sha256(ROOT/'research/data_assisted/frozen_screen.py'),class_order=list(CLASSES),
        test_loaded=False,gpu_used=False,patient_overlap_verified=False,unknown_original_test_aliases_verified=False)
    write_json(OUT/'source_manifest.json',source)
    vp={};vm={};em={};y=train.label.to_numpy()
    for name in NAMES:
        xfit=xs['train'] if name==NAMES[0] else np.concatenate([xs['train'],ex[extfit]])
        yfit=y if name==NAMES[0] else np.concatenate([y,ey[extfit]])
        head=fit(xfit,yfit,'s66_'+name,fp32=True)
        with threadpool_limits(limits=4):p=head.predict_proba(xs['val']);ep=head.predict_proba(ex[extheld])
        vm[name]=evaluate(val,p,OUT/'s66'/name,f'S66 {name} | exploratory frozen features','validation');vp[name]=p
        em[name]=evaluate(ext.iloc[extheld],ep,OUT/'s66'/name/'external_preflight',f'S66 {name} | external preflight domain check','external_preflight')
        logging.info('%s validation accuracy %.6f macroF1 %.6f; external preflight accuracy %.6f',name,vm[name]['accuracy'],vm[name]['macro_f1'],em[name]['accuracy'])
    baseline=json.loads((ROOT/'results/short_screening/s64_paired_preprocessing_cpu/square224/validation_metrics.json').read_text())
    assert abs(vm[NAMES[0]]['accuracy']-baseline['accuracy'])<1e-12
    assert abs(vm[NAMES[0]]['macro_f1']-baseline['macro_f1'])<1e-12
    delta=paired(val,vp[NAMES[0]],vp[NAMES[1]],vm[NAMES[0]],vm[NAMES[1]],OUT/'s66')
    s66gate=delta['accuracy_delta']>=.01-1e-12 and delta['macro_f1_delta']>=.01-1e-12 and delta['melanoma_recall_candidate']>=delta['melanoma_recall_control']-1e-12
    comparison_figures([dict(display_name=n,accuracy=m['accuracy'],macro_f1=m['macro_f1']) for n,m in vm.items()],OUT/'s66/comparison_figures','S66 fixed external-data screen: exploratory validation')
    groups=train.lesion_id.to_numpy();ps={n:np.full((len(train),7),np.nan) for n in NAMES};assign=np.full(len(train),-1);foldrows=[]
    for fold,(fitidx,heldidx) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(xs['train'],y,groups),1):
        assert not set(groups[fitidx])&set(groups[heldidx]);assert (assign[heldidx]==-1).all();assign[heldidx]=fold
        row=dict(fold=fold,ham_fit_images=len(fitidx),ham_held_images=len(heldidx),external_fit_images=len(extfit),lesion_overlap=0)
        for name in NAMES:
            xx=xs['train'][fitidx] if name==NAMES[0] else np.concatenate([xs['train'][fitidx],ex[extfit]])
            yy=y[fitidx] if name==NAMES[0] else np.concatenate([y[fitidx],ey[extfit]])
            head=fit(xx,yy,f's67_{name}_fold{fold}')
            with threadpool_limits(limits=4):p=head.predict_proba(xs['train'][heldidx])
            ps[name][heldidx]=p;m=metric_report(y[heldidx],p,0.)
            row[name+'_accuracy']=m['accuracy'];row[name+'_macro_f1']=m['macro_f1']
        row['net_correct_change']=int((ps[NAMES[1]][heldidx].argmax(1)==y[heldidx]).sum()-(ps[NAMES[0]][heldidx].argmax(1)==y[heldidx]).sum())
        foldrows.append(row);logging.info('Group fold %s',json.dumps(row))
    assert (assign>0).all();write_csv(OUT/'s67/fold_comparison.csv',foldrows)
    write_csv(OUT/'s67/fold_assignments.csv',pd.DataFrame(dict(image_id=train.image_id,lesion_id=groups,meta_fold=assign)).to_dict('records'))
    gm={n:evaluate(train,p,OUT/'s67'/n,f'S67 {n} | train-only lesion-group OOF','training_meta_oof') for n,p in ps.items()}
    gd=paired(train,ps[NAMES[0]],ps[NAMES[1]],gm[NAMES[0]],gm[NAMES[1]],OUT/'s67');positive=sum(r['net_correct_change']>0 for r in foldrows)
    s67gate=gd['accuracy_delta']>=.01-1e-12 and gd['macro_f1_delta']>=-1e-12 and gd['melanoma_recall_candidate']>=gd['melanoma_recall_control']-1e-12 and positive>=2
    comparison_figures([dict(display_name=n,accuracy=m['accuracy'],macro_f1=m['macro_f1']) for n,m in gm.items()],OUT/'s67/comparison_figures','S67 train-only lesion-group consistency: external training data')
    assert sha256(WEIGHTS)==wh and sha256(manifest)==manifest_hash
    write_json(OUT/'verification.json',dict(status='passed',ham_train_images=len(train),ham_validation_images=len(val),
        external_train_images=len(extfit),external_preflight_images=len(extheld),same_pretrained_weights=True,
        historical_control_metrics_reproduced=True,all_oof_images_covered_once=True,fold_local_scaling_and_weights=True,
        external_heldout_never_fit=True,source_manifest_unchanged=True,metrics_and_confusion_recomputed=True,
        test_loaded=False,gpu_used=False,training_epochs=0))
    for study,metrics,stem,protocol in [('s66',vm,'validation',c['protocol']),('s67',gm,'training_meta_oof','training_only_lesion_group_3fold')]:
        for name,m in metrics.items():
            retry_registry_upsert(dict(experiment_id=f'{study}_{name}_frozen_data_seed42',era='structured',
                record_kind='frozen_feature_data_screen',phase='post_test_external_data_development',protocol=protocol,
                evaluation_split='validation' if study=='s66' else 'training_meta_oof',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
                model='convnext_tiny',method='CPU FP32 frozen ImageNet1k; fixed weighted logistic regression C1; '+name,
                epochs=0,seed=42,status='completed',decision='data_gate_passed' if s66gate and s67gate else 'data_gate_failed',
                metrics_path=relative(OUT/study/name/(stem+'_metrics.json')),plots_dir=relative(OUT/study/name/'figures'),
                notes='Data hypothesis probe, not fine-tuned ensemble/test accuracy. External AK subset mapping; independence limits recorded.',
                **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    result=dict(status='completed',s66=dict(metrics=vm,external_preflight_metrics=em,gate_passed=bool(s66gate),**delta),
        s67=dict(metrics=gm,positive_folds=positive,gate_passed=bool(s67gate),**gd),
        data_gate_passed=bool(s66gate and s67gate),runtime_seconds=time.perf_counter()-start,test_loaded=False,gpu_used=False,
        training_epochs=0,ensemble_reference_unchanged_accuracy=.936127744510978,
        decision='prepare_bounded_gpu_proposal_only' if s66gate and s67gate else 'do_not_train_from_this_screen')
    write_json(OUT/'summary.json',result)
    logging.info('COMPLETED data gate passed=%s; S66 accuracy %.6f->%.6f; S67 accuracy %.6f->%.6f',result['data_gate_passed'],vm[NAMES[0]]['accuracy'],vm[NAMES[1]]['accuracy'],gm[NAMES[0]]['accuracy'],gm[NAMES[1]]['accuracy'])


if __name__=='__main__':
    with run_lock('s66_s67_frozen_data_screen'):main()
