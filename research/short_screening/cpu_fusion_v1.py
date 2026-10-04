"""Bounded saved-prediction fusion study; never loads an image or a test label."""
import json,warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from research.common import ROOT,CLASSES,write_json,write_csv,atomic_text,sha256,relative
from research.plots import metric_figures,comparison_figures
from research.aggressive.core import weighted_metrics,predictions,retry_registry_upsert
from research.registry import FIELDS

OUT=ROOT/'results/short_screening/cpu_fusion_v1'
PLAN=dict(version='cpu_fusion_v1',scope='post-test development diagnostic; no new untouched-test result',
    seed=42,folds=5,fold_method='StratifiedGroupKFold, lesion_id, shuffled',
    methods=['fixed_equal_probability_reference','fixed_equal_log_probability',
             'crossfit_regularized_class_bias','crossfit_regularized_probability_stacking'],
    class_bias_l2=.05,class_bias_bounds=[-2,2],stacking_C=1.0,
    meta_fit_criterion='unweighted cross-entropy; no class-weight/threshold/grid search',
    minimum_material_accuracy_gain=.01,macro_f1_must_not_decline=True,
    melanoma_and_bkl_recall_must_not_decline=True,positive_gain_required_in_at_least_folds=4,
    log_probability_floor=1e-12,
    limitations='Base checkpoints were previously selected using this validation cohort. Cross-fitting protects only new meta-fitting; it cannot undo prior selection optimism.',
    locked_test_loaded=False,gpu_used=False,cnn_training=False)

def bias_fit(p,y):
    logp=np.log(np.maximum(p,1e-12));one=np.eye(7)[y];lam=.05
    def objective(b):
        q=softmax(logp+b,axis=1)
        loss=float(-np.log(q[np.arange(len(y)),y]).mean()+.5*lam*np.dot(b,b))
        grad=(q-one).mean(axis=0)+lam*b
        return loss,grad
    r=minimize(objective,np.zeros(7),jac=True,method='L-BFGS-B',bounds=[(-2,2)]*7,options={'maxiter':200,'ftol':1e-12})
    if not r.success:raise RuntimeError('Class-bias fit did not converge: '+r.message)
    return r.x

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=OUT/'PREDECLARED_PLAN.json'
    if plan.exists():
        if json.loads(plan.read_text())!=PLAN:raise RuntimeError('Declared study changed')
    else:write_json(plan,PLAN)
    directories=['s28_efficientnet_b0_final_strict_seed42','s29_convnext_tiny_final_strict_seed42','s30_efficientnet_v2_s_final_strict_seed42']
    paths=[ROOT/'results/structured_experiments'/d/'validation_predictions.csv' for d in directories]
    paths+=[ROOT/'results/final_strict/v1/ensemble/validation_predictions.csv']
    before={relative(p):sha256(p) for p in paths}
    frames=[pd.read_csv(p) for p in paths];reference=frames[-1]
    assert len(reference)==1503 and not reference.image_id.duplicated().any()
    arrays=[]
    for frame in frames[:-1]:
        assert len(frame)==1503 and not frame.image_id.duplicated().any()
        assert set(frame.image_id)==set(reference.image_id)
        frame=frame.set_index('image_id').loc[reference.image_id].reset_index()
        assert frame.true_class.tolist()==reference.true_class.tolist()
        p=frame[[f'p_{c}' for c in CLASSES]].to_numpy(dtype=float)
        assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
        arrays.append(p)
    base=sum(arrays)/3
    np.testing.assert_allclose(base,reference[[f'p_{c}' for c in CLASSES]],atol=1e-7,rtol=1e-7)
    y=np.array([CLASSES.index(c) for c in reference.true_class]);correct=base.argmax(1)==y
    identities=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','lesion_id','split'])
    identity=identities.loc[identities.split=='val'].set_index('image_id').loc[reference.image_id]
    groups=identity.lesion_id.to_numpy()
    splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(base,y,groups))
    fold_id=np.full(len(y),-1);checks=[]
    for fold,(train,hold) in enumerate(splits):
        assert not set(groups[train])&set(groups[hold]);assert len(set(y[train]))==7
        assert (fold_id[hold]==-1).all();fold_id[hold]=fold
        checks.append(dict(fold=fold,meta_train_images=len(train),heldout_images=len(hold),
            lesion_overlap=0,heldout_class_support={c:int((y[hold]==i).sum()) for i,c in enumerate(CLASSES)}))
    assert (fold_id>=0).all()
    write_json(OUT/'provenance.json',dict(source_hashes=before,class_order=list(CLASSES),images=len(y),
        unique_validation_lesions=len(set(groups)),fold_checks=checks,
        feature_probability_floor_count=int(sum((p<1e-12).sum() for p in arrays)),**PLAN))
    write_csv(OUT/'fold_assignments.csv',[dict(image_id=i,lesion_id=g,meta_holdout_fold=int(f)) for i,g,f in zip(reference.image_id,groups,fold_id)])
    geometric=softmax(sum(np.log(np.maximum(p,1e-12)) for p in arrays)/3,axis=1)
    bias_oof=np.full_like(base,np.nan);stack_oof=np.full_like(base,np.nan)
    features=np.concatenate(arrays,axis=1)
    with threadpool_limits(limits=2):
        for fold,(train,hold) in enumerate(splits):
            bias=bias_fit(base[train],y[train]);bias_oof[hold]=softmax(np.log(np.maximum(base[hold],1e-12))+bias,axis=1)
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,random_state=42))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);model.fit(features[train],y[train])
            assert model[-1].classes_.tolist()==list(range(7));stack_oof[hold]=model.predict_proba(features[hold])
            write_json(OUT/f'fold_{fold}_meta_parameters.json',dict(fold=fold,bias=bias.tolist(),stacking_coefficients=model[-1].coef_.tolist(),
                stacking_intercepts=model[-1].intercept_.tolist(),feature_mean=model[0].mean_.tolist(),feature_scale=model[0].scale_.tolist()))
    outputs=[(PLAN['methods'][0],base),(PLAN['methods'][1],geometric),(PLAN['methods'][2],bias_oof),(PLAN['methods'][3],stack_oof)]
    rows=[];fold_rows=[];baseline_metrics=None
    for name,p in outputs:
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-7)
        m=weighted_metrics(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))
        m['loss_definition']='Unweighted cross-entropy of saved or out-of-fold probabilities'
        folder=OUT/name;write_json(folder/'validation_metrics.json',m)
        frame=predictions(reference.image_id,y,p)
        for r,f in zip(frame,fold_id):r['meta_holdout_fold']=int(f)
        write_csv(folder/'validation_predictions.csv',frame)
        metric_figures(m,folder/'figures',name+' | post-test development CPU screening')
        guessed=p.argmax(1);gains=[]
        for fold,(_,hold) in enumerate(splits):
            gain=float((guessed[hold]==y[hold]).mean()-correct[hold].mean());gains.append(gain)
            fold_rows.append(dict(method=name,fold=fold,accuracy=float((guessed[hold]==y[hold]).mean()),reference_accuracy=float(correct[hold].mean()),accuracy_gain= gain))
        if baseline_metrics is None:baseline_metrics=m
        passed=bool(m['accuracy']-baseline_metrics['accuracy']>=.01 and m['macro_f1']>=baseline_metrics['macro_f1']
            and all(m['per_class'][cl]['recall']>=baseline_metrics['per_class'][cl]['recall'] for cl in ['mel','bkl'])
            and sum(g>0 for g in gains)>=4)
        row=dict(display_name=name,accuracy=m['accuracy'],macro_f1=m['macro_f1'],mel_recall=m['per_class']['mel']['recall'],bkl_recall=m['per_class']['bkl']['recall'],
            gain_pp=100*(m['accuracy']-baseline_metrics['accuracy']),correct_gained=int(((~correct)&(guessed==y)).sum()),correct_lost=int((correct&(guessed!=y)).sum()),
            positive_folds=sum(g>0 for g in gains),material_gate_passed=passed)
        rows.append(row)
        record={key:'' for key in FIELDS};record.update(experiment_id='cpu_fusion_v1_'+name,era='structured',record_kind='cpu_fusion_screen',
            phase='short_screening',protocol='post_test_lesion_group_crossfit_validation',evaluation_split='development_validation',status='completed',
            model='frozen_strict_B0_ConvNeXtTiny_EfficientNetV2S_probabilities',method=name,seed=42,metrics_path=relative(folder/'validation_metrics.json'),
            plots_dir=relative(folder/'figures'),confusion_matrix_path=relative(folder/'figures/confusion_matrix.csv'),config_path=relative(plan),
            decision='promising_cpu_gate' if passed else 'no_material_gate',notes=PLAN['limitations'],
            **{key:m[key] for key in ['accuracy','macro_precision','macro_recall','macro_f1']})
        retry_registry_upsert(record);write_json(folder/'record.json',record)
    oracle=np.any(np.stack([p.argmax(1)==y for p in arrays]),axis=0)
    shared_by_class=[dict(class_name=cl,support=int((y==j).sum()),reference_errors=int(((y==j)&~correct).sum()),
        all_members_wrong=int(((y==j)&~oracle).sum()),reference_errors_recoverable=int(((y==j)&~correct&oracle).sum())) for j,cl in enumerate(CLASSES)]
    write_csv(OUT/'shared_error_classes.csv',shared_by_class)
    complement=[]
    for name,p in zip(directories,arrays):
        for j,cl in enumerate(CLASSES):
            mask=y==j
            complement.append(dict(model=name,class_name=cl,reference_errors_fixed=int((mask&~correct&(p.argmax(1)==y)).sum()),
                reference_correct_lost=int((mask&correct&(p.argmax(1)!=y)).sum())))
    write_csv(OUT/'class_complementarity.csv',complement);write_csv(OUT/'comparison.csv',rows);write_csv(OUT/'fold_comparison.csv',fold_rows)
    comparison_figures(rows,OUT/'figures','CPU screening | same cohort, cross-fitted meta methods')
    summary=dict(comparisons=rows,oracle_accuracy=float(oracle.mean()),reference_errors_recoverable_by_some_member=int((~correct&oracle).sum()),
        errors_all_three_members_share=int((~oracle).sum()),oracle_is_unattainable_label_informed_diagnostic=True,
        promising_methods=[r['display_name'] for r in rows if r['material_gate_passed']],no_gpu_launched=True,no_test_evaluation=True)
    write_json(OUT/'summary.json',summary)
    text='# CPU-only fusion screening\n\nSaved strict-validation probabilities only. Five lesion-group folds for class-bias and stacking fits; zero meta-train/holdout lesion overlap. Frozen checkpoints previously selected on this same validation cohort, so these are post-test development findings, not independent test estimates. No grid search, model training or test inference.\n\n'
    text+='| Method | Accuracy | Macro-F1 | MEL recall | BKL recall | Gain (pp) | Gained/lost | Positive folds | Material gate |\n|---|---:|---:|---:|---:|---:|---|---:|---|\n'
    for r in rows:text+=f'| {r["display_name"]} | {100*r["accuracy"]:.4f}% | {r["macro_f1"]:.6f} | {r["mel_recall"]:.4f} | {r["bkl_recall"]:.4f} | {r["gain_pp"]:+.4f} | {r["correct_gained"]}/{r["correct_lost"]} | {r["positive_folds"]}/5 | {r["material_gate_passed"]} |\n'
    text+=f'\nOracle: {100*summary["oracle_accuracy"]:.2f}% if labels were used to choose a correct constituent on every sample. This is not a deployable method or achieved score. {summary["reference_errors_recoverable_by_some_member"]} reference errors have at least one correct constituent; {summary["errors_all_three_members_share"]} images are wrong for every member.\n\n'
    text+='Gate was written before outputs: >=1 percentage point accuracy gain, no macro-F1/MEL/BKL recall decline and positive accuracy gain on >=4/5 folds. No extra trials will be added to rescue a failed gate. No final calibrator fitted on the complete evaluation cohort for reporting.\n'
    text+=('A gate passed: preserve as a promising development inference candidate; independent confirmation remains necessary. A GPU training run is not needed merely to apply this CPU fusion.\n' if summary['promising_methods'] else 'No gate passed: keep the proven reference. These outputs do not justify a GPU run to fit the same fusion or another random backbone.\n')
    atomic_text(ROOT/'research/short_screening/CPU_FUSION_V1_RESULTS.md',text)
    assert before=={relative(p):sha256(p) for p in paths}
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
