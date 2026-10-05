"""Fixed three-fold train-only geometry screen confirmation, using CPU features."""
import json,time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv
from research.strict_train import metric_report,run_lock
from research.short_screening.convnextv2_transfer import config,development
from research.aggressive.core import predictions,retry_registry_upsert
from research.plots import metric_figures,comparison_figures

OUT=ROOT/'results/short_screening/s65_preprocessing_train_group_cv'
CACHE=ROOT/'.cache/s64_paired_preprocessing_cpu'
NAMES=['square224','official236_crop224']


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(folds=3,seed=42,splitter='StratifiedGroupKFold shuffled lesion_id groups on original training partition only',
        models=NAMES,head='Fixed fold-local StandardScaler + LogisticRegression C1 max_iter1000 seed42',
        imbalance='Fold-training-only normalized sqrt inverse-frequency sample weights',
        gate='Pooled candidate accuracy >= control+0.01; macro-F1 and melanoma recall nondecreasing; positive net correct in at least2/3 folds',
        limit='One fixed paired comparison; no head/transform selection or hyperparameter search',
        purpose='Cheap consistency evidence before any GPU training proposal; not final CNN/ensemble performance',
        heldout_validation_used_for_fit=False,test_loaded=False,gpu_used=False)
    path=OUT/'PREDECLARED_PLAN.json'
    if path.exists(): assert json.loads(path.read_text())==plan
    else: write_json(path,plan)


def main():
    prepare()
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    c=config();train,_,_=development(c)
    source=json.loads((ROOT/'results/short_screening/s64_paired_preprocessing_cpu/source_manifest.json').read_text())
    head_dir=ROOT/'.cache/s65_preprocessing_train_group_cv';head_dir.mkdir(parents=True,exist_ok=True)
    # Only training identities, labels and features enter this study.
    y=train.label.to_numpy();groups=train.lesion_id.to_numpy();xs={}
    for name in NAMES:
        path=CACHE/f'{name}_train.npz'
        with np.load(path,allow_pickle=False) as f:
            assert f['ids'].tolist()==train.image_id.tolist() and np.array_equal(f['y'],y)
            assert str(f['weights_sha256'])==source['weights_sha256']
            xs[name]=f['features'].copy()
        assert xs[name].shape==(len(train),768) and np.isfinite(xs[name]).all()
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(xs[NAMES[0]],y,groups))
    ps={n:np.full((len(train),7),np.nan) for n in NAMES};coverage=np.zeros(len(train),int)
    assignments=np.full(len(train),-1);records=[];start=time.perf_counter()
    for fold,(fit,held) in enumerate(folds,1):
        assert not set(groups[fit])&set(groups[held]);coverage[held]+=1;assignments[held]=fold
        counts=np.bincount(y[fit],minlength=7);assert (counts>0).all()
        weights=np.sqrt(len(fit)/(7*counts));weights/=weights.mean()
        row=dict(fold=fold,fit_images=len(fit),heldout_images=len(held),lesion_overlap=0)
        for name in NAMES:
            head=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
            with threadpool_limits(limits=4):
                head.fit(xs[name][fit],y[fit],logisticregression__sample_weight=weights[y[fit]])
                assert head[-1].n_iter_.max()<1000 and head[-1].classes_.tolist()==list(range(7))
                p=head.predict_proba(xs[name][held])
            assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
            import joblib
            joblib.dump(head,head_dir/f'{name}_fold{fold}_head.joblib')
            ps[name][held]=p
            m=metric_report(y[held],p,float(-np.log(np.maximum(p[np.arange(len(held)),y[held]],1e-12)).mean()))
            for k in ['accuracy','macro_f1','macro_recall']:row[name+'_'+k]=m[k]
        row['net_correct_change']=int(((ps[NAMES[1]][held].argmax(1)==y[held]).sum())-((ps[NAMES[0]][held].argmax(1)==y[held]).sum()))
        records.append(row);print(json.dumps(row),flush=True)
    assert (coverage==1).all();write_csv(OUT/'fold_comparison.csv',records)
    write_csv(OUT/'fold_assignments.csv',pd.DataFrame(dict(image_id=train.image_id,lesion_id=groups,meta_fold=assignments)).to_dict('records'))
    metrics={}
    for name,p in ps.items():
        assert np.isfinite(p).all()
        m=metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))
        target=OUT/name;write_json(target/'training_meta_oof_metrics.json',m)
        rows=predictions(train.image_id,y,p);write_csv(target/'training_meta_oof_predictions.csv',rows)
        write_csv(target/'training_meta_oof_probabilities.csv',pd.DataFrame(rows)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
        metric_figures(m,target/'figures',f'S65 {name} | train-only lesion-group OOF')
        saved=pd.read_csv(target/'training_meta_oof_predictions.csv')
        assert saved.image_id.tolist()==train.image_id.tolist()
        rem=metric_report(y,saved[[f'p_{cl}' for cl in CLASSES]].to_numpy(),m['loss'])
        assert rem['confusion_matrix']==m['confusion_matrix'] and abs(rem['macro_f1']-m['macro_f1'])<1e-12
        metrics[name]=m
    a=metrics[NAMES[0]];b=metrics[NAMES[1]];i=list(CLASSES).index('mel')
    def mel_recall(m):
        cm=np.asarray(m['confusion_matrix']);return float(cm[i,i]/cm[i].sum())
    positive=sum(r['net_correct_change']>0 for r in records)
    passed=b['accuracy']>=a['accuracy']+.01-1e-12 and b['macro_f1']>=a['macro_f1']-1e-12 and mel_recall(b)>=mel_recall(a)-1e-12 and positive>=2
    ac=ps[NAMES[0]].argmax(1)==y;bc=ps[NAMES[1]].argmax(1)==y
    comparison_figures([dict(display_name=k,accuracy=m['accuracy'],macro_f1=m['macro_f1']) for k,m in metrics.items()],OUT/'comparison_figures','S65 train-only lesion-group consistency check')
    write_json(OUT/'verification.json',dict(status='passed',train_images=len(train),heldout_coverage=int(coverage.sum()),
        lesion_overlap=0,fold_local_scaling_and_weights=True,metrics_and_confusion_recomputed=True,test_loaded=False,gpu_used=False))
    summary=dict(status='completed',gate_passed=bool(passed),metrics=metrics,positive_folds=positive,
        gained=int((~ac&bc).sum()),lost=int((ac&~bc).sum()),net_correct_change=int(bc.sum()-ac.sum()),
        melanoma_recall_control=mel_recall(a),melanoma_recall_candidate=mel_recall(b),runtime_seconds=time.perf_counter()-start,
        test_loaded=False,gpu_used=False,training_epochs=0)
    write_json(OUT/'source_manifest.json',dict(split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
        feature_files={relative(CACHE/f'{n}_train.npz'):sha256(CACHE/f'{n}_train.npz') for n in NAMES},class_order=list(CLASSES)))
    for name,m in metrics.items():
        retry_registry_upsert(dict(experiment_id=f's65_{name}_train_group_cv_seed42',era='structured',record_kind='frozen_feature_group_cv',
            phase='post_test_development_preflight',protocol='training_only_lesion_group_3fold',evaluation_split='training_meta_oof',
            split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],model='convnext_tiny',
            method='Frozen pretrained CPU features; fixed3fold train-only grouped head screening; '+name,
            epochs=0,seed=42,status='completed',decision='gate_passed' if passed else 'gate_failed',
            metrics_path=relative(OUT/name/'training_meta_oof_metrics.json'),plots_dir=relative(OUT/name/'figures'),
            notes='Diagnostic head CV, not standalone fine-tuned CNN, final strict validation or locked-test result.',
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    write_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    import sys
    if '--prepare' in sys.argv:prepare()
    else:
        with run_lock('s65_preprocessing_train_group_cv'):main()
