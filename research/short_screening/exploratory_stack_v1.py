"""One CPU-only cross-fitted learned fusion on existing exploratory probabilities."""
import json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from research.common import ROOT, CLASSES, sha256, relative, write_json, write_csv
from research.strict_train import metric_report
from research.aggressive.core import predictions, retry_registry_upsert
from research.plots import metric_figures, comparison_figures

OUT=ROOT/'results/short_screening/s42_exploratory_stack_v1'
SPLIT='data/splits/exploratory/image_level_dev_v1.csv'
MEMBERS=['s06_convnext_tiny_none_exploratory_seed42','s18_convnext_small_none_exploratory_seed42',
         's15_densenet201_none_exploratory_seed42','s10_efficientnet_v2_s_none_exploratory_seed42']
PLAN=dict(protocol='exploratory_image_level',scope='post-test development; cross-fit protects only new meta-fitting',
    members=MEMBERS,methods=['equal_four_control','lesion_group_crossfit_centered_logprob_stacking'],
    folds=5,seed=42,stacking_C=1.,solver='lbfgs',max_iter=2000,class_weight=None,
    probability_floor=1e-8,feature_definition='Concatenate within-model centered log probabilities; fold-train StandardScaler',
    hyperparameter_search=False,source_checkpoint_selection='Preserved accuracy winners',
    material_gate='At least +.005 accuracy over S27; F1 >= frozen .877454; positive vs S27 on >=4/5 meta holdout folds',
    caveat='Base CNN checkpoints were selected on this validation cohort and exploratory training can share lesions with it. This is not independent CNN cross-validation or test performance.',
    gpu_used=False,test_loaded=False)


def main():
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=False);write_json(OUT/'PREDECLARED_PLAN.json',PLAN)
    identities=pd.read_csv(ROOT/SPLIT,usecols=['image_id','lesion_id','split'])
    excluded=set((identities.index[identities.split!='val']+1).tolist())
    val=pd.read_csv(ROOT/SPLIT,skiprows=lambda line:line in excluded)
    assert len(val)==1503 and set(val.split)=={'val'} and val.image_id.is_unique
    locked=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','split'])
    assert not set(val.image_id)&set(locked.loc[locked.split=='test','image_id'])
    arrays=[];hashes={}
    for rid in MEMBERS+['s27_final_weighted_02_exploratory_seed42']:
        path=ROOT/'results/structured_experiments'/rid/'validation_predictions.csv'
        frame=pd.read_csv(path);hashes[relative(path)]=sha256(path)
        assert frame.image_id.is_unique and set(frame.image_id)==set(val.image_id)
        frame=frame.set_index('image_id').loc[val.image_id]
        assert frame.true_class.tolist()==val.diagnosis.tolist()
        p=frame[[f'p_{cl}' for cl in CLASSES]].to_numpy()
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        arrays.append(p)
    best=arrays.pop();base=np.mean(arrays,axis=0);y=val.label.to_numpy();groups=val.lesion_id.to_numpy()
    features=[]
    for p in arrays:
        logp=np.log(np.maximum(p,1e-8));features.append(logp-logp.mean(1,keepdims=True))
    x=np.concatenate(features,axis=1);oof=np.full_like(base,np.nan);fold_ids=np.full(len(y),-1);checks=[]
    with threadpool_limits(limits=2):
        for fold,(train,hold) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(x,y,groups)):
            assert not set(groups[train])&set(groups[hold]) and len(set(y[train]))==7
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,random_state=42))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);model.fit(x[train],y[train])
            assert model[-1].classes_.tolist()==list(range(7));oof[hold]=model.predict_proba(x[hold]);fold_ids[hold]=fold
            write_json(OUT/f'fold_{fold}_parameters.json',dict(scaler_mean=model[0].mean_.tolist(),scaler_scale=model[0].scale_.tolist(),coefficients=model[-1].coef_.tolist(),intercept=model[-1].intercept_.tolist()))
            checks.append(dict(fold=fold,meta_train_images=len(train),meta_holdout_images=len(hold),lesion_overlap=0))
    assert np.isfinite(oof).all() and np.allclose(oof.sum(1),1,atol=1e-7) and (fold_ids>=0).all()
    write_csv(OUT/'fold_assignments.csv',[dict(image_id=i,lesion_id=g,meta_holdout_fold=int(f)) for i,g,f in zip(val.image_id,groups,fold_ids)])
    write_json(OUT/'provenance.json',dict(source_prediction_sha256=hashes,split_sha256=sha256(ROOT/SPLIT),class_order=CLASSES,fold_checks=checks,test_loaded=False,gpu_used=False))
    rows=[];metrics={};baseline_correct=best.argmax(1)==y;fold_rows=[]
    for name,p in [('historical_s27_accuracy_reference',best),('equal_four_control',base),('crossfit_stacking',oof)]:
        folder=OUT/name;m=metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()));metrics[name]=m
        write_json(folder/'validation_metrics.json',m);pred=predictions(val.image_id,y,p)
        for row,fold in zip(pred,fold_ids):row['meta_holdout_fold']=int(fold)
        write_csv(folder/'validation_predictions.csv',pred);write_csv(folder/'validation_probabilities.csv',pd.DataFrame(pred)[['image_id']+[f'p_{c}' for c in CLASSES]].to_dict('records'))
        metric_figures(m,folder/'figures',f'S42 {name} | exploratory development')
        rows.append(dict(display_name=name,**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
        for fold in range(5):
            hold=fold_ids==fold;gain=float(np.mean(p.argmax(1)[hold]==y[hold])-baseline_correct[hold].mean())
            fold_rows.append(dict(method=name,fold=fold,images=int(hold.sum()),gain_over_s27_pp=100*gain))
        if name!='historical_s27_accuracy_reference':
            retry_registry_upsert(dict(experiment_id='s42_'+name+'_exploratory_seed42',era='structured',record_kind='cpu_fusion_screen',phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation_meta_oof' if name=='crossfit_stacking' else 'validation',split_manifest=SPLIT,split_sha256=sha256(ROOT/SPLIT),method=name,ensemble_members=json.dumps(MEMBERS),epochs=0,seed=42,metrics_path=relative(folder/'validation_metrics.json'),plots_dir=relative(folder/'figures'),status='completed',notes=PLAN['caveat'],**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    comparison_figures(rows,OUT/'comparison_figures','S42 saved exploratory predictions; bounded CPU study')
    write_csv(OUT/'fold_comparison.csv',fold_rows)
    candidates={}
    for name,p in [('equal_four_control',base),('crossfit_stacking',oof)]:
        correct=p.argmax(1)==y;gained=int((~baseline_correct&correct).sum());lost=int((baseline_correct&~correct).sum())
        positive=sum(r['gain_over_s27_pp']>0 for r in fold_rows if r['method']==name)
        passed=metrics[name]['accuracy']-metrics['historical_s27_accuracy_reference']['accuracy']>=.005-1e-12 and metrics[name]['macro_f1']>=.8774541331925992 and positive>=4
        candidates[name]=dict(gained=gained,lost=lost,net=gained-lost,positive_folds=positive,material_gate_passed=bool(passed))
    for path,digest in hashes.items():assert sha256(ROOT/path)==digest
    summary=dict(status='completed',methods=rows,candidate_decisions=candidates,source_artifacts_unchanged=True,limitations=PLAN['caveat'],gpu_used=False,test_loaded=False,decision='No C/weight/cohort search or automatic GPU training; stop after bounded study')
    write_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
