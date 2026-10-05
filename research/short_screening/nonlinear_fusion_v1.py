"""One fixed low-capacity nonlinear fusion check with lesion-group cross-fitting."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,f1_score
from threadpoolctl import threadpool_limits
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv
from research.strict_train import metric_report
from research.aggressive.core import predictions,retry_registry_upsert
from research.plots import metric_figures,comparison_figures

OUT=ROOT/'results/short_screening/s49_nonlinear_fusion_v1'
PARENT=ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion'
PARAMS=dict(loss='log_loss',learning_rate=.05,max_iter=100,max_leaf_nodes=7,
            min_samples_leaf=30,l2_regularization=10.,early_stopping=False,random_state=42)
PLAN=dict(model='HistGradientBoostingClassifier',parameters=PARAMS,features='Concatenated four model probability vectors; no images or new embeddings',
    folds=5,fold_method='StratifiedGroupKFold; lesion groups; shuffled seed42',
    scope='Post-test exploratory development; CNNs previously selected on this full validation cohort',
    gate='>=+.005 accuracy over S46; macro-F1 and MEL recall nondecrease; positive accuracy gain in >=4/5 heldout meta folds',
    search=False,budget='One configuration, five meta folds, no follow-up hyperparameter tuning',
    deployment='No whole-validation final fit; reported probabilities are heldout-meta outputs, not an independently tested deployable model',
    test_loaded=False,gpu_used=False)


def main():
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=False);write_json(OUT/'PREDECLARED_PLAN.json',PLAN)
    manifest=json.loads((PARENT/'candidate_manifest.json').read_text())
    reference=pd.read_csv(PARENT/'validation_predictions.csv');assert len(reference)==1503 and reference.image_id.is_unique
    split='data/splits/exploratory/image_level_dev_v1.csv'
    identities=pd.read_csv(ROOT/split,usecols=['image_id','lesion_id','split'])
    assert set(reference.image_id)==set(identities.loc[identities.split=='val','image_id'])
    locked=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','split'])
    assert not set(reference.image_id)&set(locked.loc[locked.split=='test','image_id'])
    groups=identities.set_index('image_id').loc[reference.image_id].lesion_id.to_numpy();y=np.array([CLASSES.index(c) for c in reference.true_class])
    arrays=[]
    for source in manifest['sources']:
        path=ROOT/source['prediction_file'];assert sha256(path)==source['prediction_sha256']
        frame=pd.read_csv(path);assert frame.image_id.is_unique and set(frame.image_id)==set(reference.image_id)
        frame=frame.set_index('image_id').loc[reference.image_id]
        assert frame.true_class.tolist()==reference.true_class.tolist()
        arrays.append(frame[[f'p_{cl}' for cl in CLASSES]].to_numpy())
    x=np.concatenate(arrays,axis=1);assert np.isfinite(x).all()
    baseline=np.mean(arrays,axis=0);np.testing.assert_allclose(baseline,reference[[f'p_{cl}' for cl in CLASSES]],atol=1e-12)
    oof=np.full_like(baseline,np.nan);fold_ids=np.full(len(y),-1);fold_rows=[]
    with threadpool_limits(limits=2):
        for fold,(train,hold) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(x,y,groups)):
            assert not set(groups[train])&set(groups[hold]) and len(set(y[train]))==7
            assert (fold_ids[hold]==-1).all()
            model=HistGradientBoostingClassifier(**PARAMS);model.fit(x[train],y[train])
            assert model.classes_.tolist()==list(range(7));oof[hold]=model.predict_proba(x[hold]);fold_ids[hold]=fold
            a=baseline[hold].argmax(1)==y[hold];b=oof[hold].argmax(1)==y[hold]
            fold_rows.append(dict(fold=fold,meta_train_images=len(train),meta_holdout_images=len(hold),lesion_overlap=0,reference_accuracy=float(a.mean()),candidate_accuracy=float(b.mean()),gain_pp=float(100*(b.mean()-a.mean()))))
    assert (fold_ids>=0).all() and np.isfinite(oof).all() and np.allclose(oof.sum(1),1,atol=1e-7)
    m=metric_report(y,oof,float(-np.log(np.maximum(oof[np.arange(len(y)),y],1e-12)).mean()))
    write_json(OUT/'validation_meta_oof_metrics.json',m);pred=predictions(reference.image_id,y,oof)
    for row,fold in zip(pred,fold_ids):row['meta_holdout_fold']=int(fold)
    write_csv(OUT/'validation_meta_oof_predictions.csv',pred)
    write_csv(OUT/'validation_meta_oof_probabilities.csv',pd.DataFrame(pred)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
    write_csv(OUT/'fold_comparison.csv',fold_rows)
    write_csv(OUT/'fold_assignments.csv',[dict(image_id=i,lesion_id=g,meta_holdout_fold=int(f)) for i,g,f in zip(reference.image_id,groups,fold_ids)])
    metric_figures(m,OUT/'figures','S49 nonlinear fusion | heldout meta folds; exploratory CNNs')
    saved=pd.read_csv(OUT/'validation_meta_oof_predictions.csv')
    assert abs(accuracy_score(saved.true_class,saved.predicted_class)-m['accuracy'])<1e-12
    assert abs(f1_score(saved.true_class,saved.predicted_class,labels=CLASSES,average='macro')-m['macro_f1'])<1e-12
    bm=json.loads((PARENT/'validation_metrics.json').read_text());a=baseline.argmax(1)==y;b=oof.argmax(1)==y
    positives=sum(r['gain_pp']>0 for r in fold_rows)
    passed=m['accuracy']-bm['accuracy']>=.005-1e-12 and m['macro_f1']>=bm['macro_f1'] and m['per_class']['mel']['recall']>=bm['per_class']['mel']['recall'] and positives>=4
    rows=[dict(display_name='S46 fixed soft voting',accuracy=bm['accuracy'],macro_f1=bm['macro_f1']),dict(display_name='S49 heldout-meta nonlinear fusion',accuracy=m['accuracy'],macro_f1=m['macro_f1'])]
    comparison_figures(rows,OUT/'comparison_figures','One bounded nonlinear fusion check')
    summary=dict(status='completed',comparisons=rows,gained=int((~a&b).sum()),lost=int((a&~b).sum()),net=int(b.sum()-a.sum()),positive_folds=positives,mel_recall=m['per_class']['mel']['recall'],material_gate_passed=bool(passed),fold_checks=fold_rows,
        source_manifest=relative(PARENT/'candidate_manifest.json'),parameters=PARAMS,limitations=PLAN['scope']+'; cross-fitting only protects the meta-model, not prior CNN/checkpoint selection.',test_loaded=False,gpu_used=False,decision='Single nonlinear configuration complete; stop, no tuning or automatic training')
    write_json(OUT/'summary.json',summary)
    retry_registry_upsert(dict(experiment_id='s49_nonlinear_crossfit_fusion_exploratory_seed42',era='structured',record_kind='cpu_fusion_screen',phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation_meta_oof',split_manifest=split,split_sha256=sha256(ROOT/split),method='Fixed low-capacity HistGradientBoosting probability fusion; lesion-group meta cross-fit',ensemble_members=json.dumps([s['run'] for s in manifest['sources']]),epochs=0,seed=42,metrics_path=relative(OUT/'validation_meta_oof_metrics.json'),plots_dir=relative(OUT/'figures'),status='completed',decision='material_gate_passed' if passed else 'material_gate_failed',notes=summary['limitations'],**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
