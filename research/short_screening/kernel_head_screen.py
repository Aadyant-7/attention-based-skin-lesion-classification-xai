"""S72: one fixed nonlinear image-feature head versus linear head, train-only.

Reuses frozen ImageNet feature caches, not fine-tuned validation-selected CNN
features. No GPU, image inference, test labels or parameter search.
"""
import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.common import ROOT,CLASSES,relative,sha256,write_csv,write_json
from research.plots import metric_figures,comparison_figures
from research.registry import upsert
from research.short_screening.lesion_bag_screen import report

OUT=ROOT/'results/short_screening/s72_nonlinear_frozen_feature_head'
CACHE=ROOT/'.cache/s72_nonlinear_frozen_feature_head'
FEATURES=ROOT/'.cache/s64_paired_preprocessing_cpu'
SPLIT=ROOT/'data/splits/exploratory/image_level_dev_v1.csv'
SPLIT_HASH='75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb'
WEIGHT_HASH='983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'
P_COLS=[f'p_{c}' for c in CLASSES]
NAMES=['linear','rbf_nystroem']
PLAN=dict(question='Are nonlinear class boundaries in existing pretrained image features useful beyond a fixed linear head?',
    input='Cached identity square224 ConvNeXt-Tiny ImageNet1k FP32 768-dimensional features; no fine-tuned CNN or S53 prediction input',
    control='Fold-local StandardScaler -> LogisticRegression C1 max_iter1000 seed42',
    candidate='Fold-local StandardScaler -> Nystroem RBF gamma1/768 512 components seed42 -> StandardScaler -> same logistic head',
    imbalance='Fold-training-only normalized sqrt inverse-frequency sample weights',
    folds='Same3 StratifiedGroupKFold shuffled lesion_id groups seed42 on7009 original development-training images only',
    gate='Pooled nonlinear accuracy >= linear+0.01; no macro-F1 or melanoma-recall decline; positive accuracy net in at least2/3 folds',
    validation_rule='Do not load development-validation feature cache or scoring labels unless train-only gate passes; one fixed paired full-train head comparison afterward',
    limitation='Diagnostic frozen-feature head, not a trained CNN/ensemble gain; consistency evidence cannot guarantee fine-tuning benefit',
    cpu_threads=4,soft_wall_budget_seconds=300,search=False,test_loaded=False,gpu_used=False,
    source='https://scikit-learn.org/stable/modules/kernel_approximation.html#nystroem-method-for-kernel-approximation')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'PREDECLARED_PLAN.json'
    if path.exists():assert json.loads(path.read_text())==PLAN
    else:write_json(path,PLAN)


def load(part, identity):
    path=FEATURES/f'square224_{part}.npz'
    with np.load(path,allow_pickle=False) as f:
        assert str(f['weights_sha256'])==WEIGHT_HASH
        ids=f['ids'].copy();y=f['y'].copy();x=f['features'].copy()
    assert len(ids)==len(set(ids)) and set(ids)==set(identity.image_id)
    frame=identity.set_index('image_id').loc[ids].reset_index()
    assert x.shape==(len(frame),768) and np.isfinite(x).all()
    assert np.isin(y,np.arange(7)).all()
    return frame,x,y,dict(path=relative(path),sha256=sha256(path),weights_sha256=WEIGHT_HASH)


def fit(name,x,y):
    head=LogisticRegression(C=1,max_iter=1000,random_state=42)
    if name=='linear':net=make_pipeline(StandardScaler(),head)
    else:net=make_pipeline(StandardScaler(),Nystroem(kernel='rbf',gamma=1/768,n_components=512,random_state=42),StandardScaler(),head)
    counts=np.bincount(y,minlength=7);assert (counts>0).all()
    w=np.sqrt(len(y)/(7*counts));w/=w.mean()
    with threadpool_limits(limits=4):net.fit(x,y,logisticregression__sample_weight=w[y])
    assert net[-1].n_iter_.max()<1000 and net[-1].classes_.tolist()==list(range(7))
    return net


def save(frame,y,p,path,stem,title):
    assert p.shape==(len(frame),7) and np.isfinite(p).all() and (p>=0).all()
    assert np.allclose(p.sum(1),1,atol=1e-7)
    m=report(y,p);write_json(path/(stem+'_metrics.json'),m)
    rows=frame.copy();rows['true_class']=[CLASSES[i] for i in y]
    rows['predicted_class']=[CLASSES[i] for i in p.argmax(1)];rows[P_COLS]=p
    write_csv(path/(stem+'_predictions.csv'),rows.to_dict('records'))
    write_csv(path/(stem+'_probabilities.csv'),rows[['image_id']+P_COLS].to_dict('records'))
    metric_figures(m,path/'figures',title)
    check=pd.read_csv(path/(stem+'_predictions.csv'))
    rem=report(check.true_class.map(dict(zip(CLASSES,range(7)))).to_numpy(),check[P_COLS].to_numpy())
    assert check.image_id.tolist()==frame.image_id.tolist()
    assert rem['confusion_matrix']==m['confusion_matrix'] and abs(rem['macro_f1']-m['macro_f1'])<1e-12
    return m


def main():
    prepare()
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    lock=OUT/'run.lock';handle=lock.open('x')
    try:
        CACHE.mkdir(parents=True,exist_ok=True)
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',encoding='utf-8'),logging.StreamHandler()])
        start=time.perf_counter();logging.info('START S72 train-only grouped nonlinear head; no images/GPU/test')
        assert sha256(SPLIT)==SPLIT_HASH
        ids=pd.read_csv(SPLIT,usecols=['image_id','lesion_id','split'])
        train_ids=ids.loc[ids.split=='train'].copy()
        original=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','lesion_id','split'])
        locked=original.loc[original.split=='test']
        assert not set(train_ids.image_id)&set(locked.image_id)
        assert not set(train_ids.lesion_id)&set(locked.lesion_id)
        train,x,y,source=load('train',train_ids);assert len(train)==7009
        groups=train.lesion_id.to_numpy()
        splits=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(x,y,groups))
        scores={name:np.full((len(y),7),np.nan) for name in NAMES}
        coverage=np.zeros(len(y),int);assign=np.full(len(y),-1,int);fold_rows=[]
        for fold,(f,h) in enumerate(splits,1):
            assert not set(groups[f])&set(groups[h]);coverage[h]+=1;assign[h]=fold
            row=dict(fold=fold,fit_images=len(f),heldout_images=len(h),lesion_overlap=0)
            for name in NAMES:
                if time.perf_counter()-start>300:raise TimeoutError('Five-minute CPU budget; preserve completed fold models, no automatic retry')
                net=fit(name,x[f],y[f]);joblib.dump(net,CACHE/f'{name}_fold{fold}.joblib')
                with threadpool_limits(limits=4):p=net.predict_proba(x[h])
                scores[name][h]=p;m=report(y[h],p)
                row[name+'_accuracy']=m['accuracy'];row[name+'_macro_f1']=m['macro_f1']
            row['net_correct']=int((scores[NAMES[1]][h].argmax(1)==y[h]).sum()-(scores[NAMES[0]][h].argmax(1)==y[h]).sum())
            fold_rows.append(row);write_csv(OUT/'fold_comparison.csv',fold_rows)
            logging.info('Fold %d linear %.4f%% nonlinear %.4f%% net %+d',fold,row['linear_accuracy']*100,row['rbf_nystroem_accuracy']*100,row['net_correct'])
        assert (coverage==1).all()
        train['meta_fold']=assign;write_csv(OUT/'fold_assignments.csv',train[['image_id','lesion_id','meta_fold']].to_dict('records'))
        metrics={name:save(train,y,p,OUT/name,'training_meta_oof',f'S72 {name} | train-only lesion-group OOF') for name,p in scores.items()}
        a,b=metrics[NAMES[0]],metrics[NAMES[1]]
        positive=sum(row['net_correct']>0 for row in fold_rows)
        passed=b['accuracy']>=a['accuracy']+.01-1e-12 and b['macro_f1']>=a['macro_f1']-1e-12 and b['per_class']['mel']['recall']>=a['per_class']['mel']['recall']-1e-12 and positive>=2
        comparison_figures([dict(display_name=name,accuracy=m['accuracy'],macro_f1=m['macro_f1']) for name,m in metrics.items()],OUT/'comparison_figures','S72 train-only grouped nonlinear feature-head screen')
        old=scores[NAMES[0]].argmax(1)==y;new=scores[NAMES[1]].argmax(1)==y
        summary=dict(status='completed',train_group_metrics=metrics,positive_folds=positive,gate_passed=bool(passed),
                     gained=int((~old&new).sum()),lost=int((old&~new).sum()),net_correct=int(new.sum()-old.sum()),
                     validation_scored=False,test_loaded=False,gpu_used=False,training_epochs=0,
                     decision='Gate passed: one fixed development validation comparison permitted' if passed else 'Reject this fixed nonlinear head; no validation scoring or GPU proposal')
        sources=[source]
        if passed:
            val_ids=ids.loc[ids.split=='val'].copy()
            assert not set(val_ids.image_id)&set(locked.image_id) and not set(val_ids.lesion_id)&set(locked.lesion_id)
            val,vx,vy,vs=load('val',val_ids);sources.append(vs)
            vm={}
            for name in NAMES:
                net=fit(name,x,y);joblib.dump(net,CACHE/f'{name}_full_train.joblib')
                with threadpool_limits(limits=4):p=net.predict_proba(vx)
                vm[name]=save(val,vy,p,OUT/name/'development_validation','validation',f'S72 {name} | exploratory frozen-feature diagnostic')
            summary.update(validation_scored=True,validation_metrics=vm)
        write_json(OUT/'source_manifest.json',dict(feature_files=sources,split_sha256=SPLIT_HASH,class_order=list(CLASSES),script_sha256=sha256(Path(__file__))))
        assert all(sha256(ROOT/s['path'])==s['sha256'] for s in sources)
        write_json(OUT/'verification.json',dict(status='passed',heldout_coverage=int(coverage.sum()),
                   fold_lesion_overlap=0,all_transforms_and_weights_fit_on_fold_training_only=True,
                   pretrained_feature_only=True,metrics_and_confusion_recomputed=True,source_files_unchanged=True,
                   validation_feature_cache_loaded=bool(passed),original_test_labels_read=False,gpu_used=False))
        for name,m in metrics.items():
            upsert(dict(experiment_id=f's72_{name}_train_group_cv_seed42',era='structured',
                        record_kind='frozen_feature_group_cv',phase='post_test_development_preflight',
                        protocol='training_only_lesion_group_3fold',evaluation_split='training_meta_oof',
                        split_manifest=relative(SPLIT),split_sha256=SPLIT_HASH,model='convnext_tiny_imagenet1k_frozen',
                        method=PLAN['control'] if name=='linear' else PLAN['candidate'],epochs=0,seed=42,status='completed',
                        decision='gate_passed' if passed else 'gate_failed',metrics_path=relative(OUT/name/'training_meta_oof_metrics.json'),
                        plots_dir=relative(OUT/name/'figures'),checkpoint=relative(CACHE/f'{name}_fold1.joblib'),
                        notes='Three saved fold heads, not a fine-tuned CNN or S53 improvement; no test/GPU. No validation scoring if grouped gate fails.',
                        **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
        summary['runtime_seconds']=time.perf_counter()-start
        write_json(OUT/'summary.json',summary);logging.info('COMPLETE gate %s runtime %.1fs',passed,summary['runtime_seconds'])
        print(json.dumps(summary,indent=2))
    finally:
        handle.close();lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else main()
