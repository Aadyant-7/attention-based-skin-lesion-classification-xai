"""One five-fold scalar-temperature calibration study on exact S53 members."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.common import ROOT,CLASSES,sha256,relative,write_json,write_csv
from research.strict_train import metric_report
from research.short_screening.panderm_fusion_v1 import aligned
from research.short_screening.error_audit_v1 import ece
from research.aggressive.core import predictions,retry_registry_upsert
from research.plots import metric_figures,comparison_figures

OUT=ROOT/'results/short_screening/s63_crossfit_temperature_calibration'
REF=ROOT/'results/short_screening/s53_equal_five_b0_addition'
FOUR=ROOT/'results/short_screening/s46_all_f1_checkpoint_fusion'
SPLIT=ROOT/'data/splits/exploratory/image_level_dev_v1.csv'
BOUNDS=(.25,10.)
PLAN=dict(question='Does calibrating each exact S53 member improve equal fusion of its existing probabilities?',
    method='One positive scalar temperature per member per meta fold; minimize natural-frequency NLL on meta-train rows only',
    folds=5,fold_method='StratifiedGroupKFold, lesion_id groups, shuffle seed42',
    optimizer='Bounded scalar minimization in log-temperature; bounds log(.25) to log(10), xatol1e-6, maxiter200',
    probability_floor=1e-12,weights=[.2]*5,
    material_gate='At least8net additional correct images over S53 AND accuracy gain>=.005 AND no macro-F1 or melanoma-recall decline',
    scope='Post-test exploratory development; base checkpoints previously selected on this whole cohort',
    deployment='Held-out-meta predictions only; no whole-validation temperature fit/deployment',
    budget='Exactly one configuration, five folds, five existing members; no alternate bounds, weights, thresholds or checkpoint selection',
    test_loaded=False,gpu_used=False)


def scaled(logp,t):
    z=logp/t
    return np.exp(z-logsumexp(z,axis=1,keepdims=True))


def nll(logp,y,t):
    z=logp/t
    return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))


def main():
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=False);write_json(OUT/'PREDECLARED_PLAN.json',PLAN)
    ref=pd.read_csv(REF/'validation_predictions.csv');assert len(ref)==1503 and ref.image_id.is_unique
    ids=ref.image_id.to_numpy();labels=ref.true_class.tolist();y=np.array([CLASSES.index(c) for c in labels])
    identities=pd.read_csv(SPLIT,usecols=['image_id','lesion_id','split'])
    assert set(ids)==set(identities.loc[identities.split=='val','image_id'])
    val=identities.set_index('image_id').loc[ids];groups=val.lesion_id.to_numpy()
    locked=pd.read_csv(ROOT/'data/splits/split_assignments.csv',usecols=['image_id','lesion_id','split'])
    test=locked.loc[locked.split=='test'];assert not set(ids)&set(test.image_id) and not set(groups)&set(test.lesion_id)
    sources=json.loads((FOUR/'candidate_manifest.json').read_text())['sources']
    bm=json.loads((REF/'validation_metrics.json').read_text());bsummary=json.loads((REF/'summary.json').read_text())
    b0='s03_efficientnet_b0_none_exploratory_seed42'
    bpath=ROOT/'results/structured_experiments'/b0/'validation_predictions.csv';bcp=ROOT/'checkpoints/structured'/b0/'best.pt'
    sources.append(dict(run=b0,prediction_file=relative(bpath),prediction_sha256=bsummary['source_prediction_sha256'][relative(bpath)],
        checkpoint=relative(bcp),checkpoint_sha256=bsummary['rescue_checkpoint_sha256']))
    arrays=[]
    for s in sources:
        assert sha256(ROOT/s['prediction_file'])==s['prediction_sha256'] and sha256(ROOT/s['checkpoint'])==s['checkpoint_sha256']
        arrays.append(aligned(ROOT/s['prediction_file'],ids,labels))
    stack=np.stack(arrays);base=stack.mean(0)
    np.testing.assert_allclose(base,aligned(REF/'validation_predictions.csv',ids,labels),atol=1e-12,rtol=0)
    logp=np.log(np.maximum(stack,1e-12))
    calibrated=np.full_like(stack,np.nan);fold_ids=np.full(len(y),-1);fit_rows=[];fold_rows=[]
    splitter=StratifiedGroupKFold(5,shuffle=True,random_state=42)
    for fold,(train,hold) in enumerate(splitter.split(base,y,groups)):
        assert not set(groups[train])&set(groups[hold]) and set(y[train])==set(range(7))
        assert (fold_ids[hold]==-1).all();fold_ids[hold]=fold
        for j,s in enumerate(sources):
            # No holdout label enters the optimization objective.
            result=minimize_scalar(lambda lt:nll(logp[j,train],y[train],np.exp(lt)),method='bounded',
                bounds=tuple(np.log(BOUNDS)),options=dict(xatol=1e-6,maxiter=200))
            assert result.success and np.isfinite(result.fun)
            t=float(np.exp(result.x));assert BOUNDS[0]<=t<=BOUNDS[1]
            q=scaled(logp[j,hold],t)
            assert np.isfinite(q).all() and np.allclose(q.sum(1),1,atol=1e-12)
            assert np.array_equal(q.argmax(1),stack[j,hold].argmax(1))
            calibrated[j,hold]=q
            fit_rows.append(dict(fold=fold,model=s['run'],temperature=t,meta_train_images=len(train),meta_holdout_images=len(hold),
                train_nll_before=nll(logp[j,train],y[train],1),train_nll_after=float(result.fun),
                holdout_nll_before=nll(logp[j,hold],y[hold],1),holdout_nll_after=nll(logp[j,hold],y[hold],t),
                objective_evaluations=int(result.nfev),optimizer_success=bool(result.success),argmax_changes=0,
                at_bound=bool(t<BOUNDS[0]*1.001 or t>BOUNDS[1]/1.001)))
        fused=calibrated[:,hold].mean(0);old=base[hold].argmax(1)==y[hold];new=fused.argmax(1)==y[hold]
        fold_rows.append(dict(fold=fold,meta_train_images=len(train),meta_holdout_images=len(hold),lesion_overlap=0,
            reference_accuracy=float(old.mean()),calibrated_accuracy=float(new.mean()),gained=int((~old&new).sum()),lost=int((old&~new).sum()),net=int(new.sum()-old.sum())))
    assert (fold_ids>=0).all() and np.isfinite(calibrated).all()
    assert np.array_equal(calibrated.argmax(2),stack.argmax(2))
    p=calibrated.mean(0)
    loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean());m=metric_report(y,p,loss)
    write_json(OUT/'validation_meta_oof_metrics.json',m)
    pred=predictions(ids,y,p)
    for row,f in zip(pred,fold_ids):row['meta_holdout_fold']=int(f)
    write_csv(OUT/'validation_meta_oof_predictions.csv',pred)
    write_csv(OUT/'validation_meta_oof_probabilities.csv',pd.DataFrame(pred)[['image_id','meta_holdout_fold']+[f'p_{c}' for c in CLASSES]].to_dict('records'))
    write_csv(OUT/'temperatures_by_fold.csv',fit_rows);write_csv(OUT/'fold_comparison.csv',fold_rows)
    write_csv(OUT/'fold_assignments.csv',[dict(image_id=i,lesion_id=g,meta_holdout_fold=int(f)) for i,g,f in zip(ids,groups,fold_ids)])
    calibrated_folder=OUT/'calibrated_members';member_rows=[]
    for j,s in enumerate(sources):
        records=predictions(ids,y,calibrated[j]);write_csv(calibrated_folder/(s['run']+'.csv'),records)
        a,b=ece(y,stack[j]);c,d=ece(y,calibrated[j])
        member_rows.append(dict(model=s['run'],nll_before=nll(logp[j],y,1),nll_after=float(-np.log(np.maximum(calibrated[j,np.arange(len(y)),y],1e-12)).mean()),ece10_before=a,ece10_after=c,argmax_changes=0))
    write_csv(OUT/'member_calibration_comparison.csv',member_rows)
    a,abins=ece(y,base);b,bbins=ece(y,p)
    write_csv(OUT/'reference_reliability.csv',abins);write_csv(OUT/'calibrated_reliability.csv',bbins)
    metric_figures(m,OUT/'figures','S63 equal calibrated fusion | held-out meta folds; exploratory CNNs')
    saved=pd.read_csv(OUT/'validation_meta_oof_predictions.csv')
    assert abs(accuracy_score(saved.true_class,saved.predicted_class)-m['accuracy'])<1e-12
    assert abs(f1_score(saved.true_class,saved.predicted_class,labels=CLASSES,average='macro')-m['macro_f1'])<1e-12
    assert np.array_equal(confusion_matrix(saved.true_class,saved.predicted_class,labels=CLASSES),m['confusion_matrix'])
    old=base.argmax(1)==y;new=p.argmax(1)==y;gained=int((~old&new).sum());lost=int((old&~new).sum());net=int(new.sum()-old.sum())
    material=net>=8 and m['accuracy']-bm['accuracy']>=.005-1e-12 and m['macro_f1']>=bm['macro_f1'] and m['per_class']['mel']['recall']>=bm['per_class']['mel']['recall']
    rows=[dict(display_name='S53 equal five reference',accuracy=bm['accuracy'],macro_f1=bm['macro_f1']),dict(display_name='S63 held-out-meta calibrated equal five',accuracy=m['accuracy'],macro_f1=m['macro_f1'])]
    comparison_figures(rows,OUT/'comparison_figures','One fixed scalar-temperature cross-fit study')
    write_csv(OUT/'class_comparison.csv',[dict(class_name=c,reference_precision=bm['per_class'][c]['precision'],candidate_precision=m['per_class'][c]['precision'],reference_recall=bm['per_class'][c]['recall'],candidate_recall=m['per_class'][c]['recall'],reference_f1=bm['per_class'][c]['f1'],candidate_f1=m['per_class'][c]['f1']) for c in CLASSES])
    fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],'--',color='gray')
    for bins,label in [(abins,'S53 reference'),(bbins,'S63 cross-fit temperatures')]:ax.plot([r['confidence'] for r in bins],[r['accuracy'] for r in bins],'o-',label=label)
    ax.set(xlabel='Mean predicted confidence',ylabel='Observed accuracy',title='Fixed ten-bin ensemble reliability');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/'comparison_figures'/f'reliability_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    for s in sources:assert sha256(ROOT/s['prediction_file'])==s['prediction_sha256']
    write_json(OUT/'source_manifest.json',dict(sources=sources,weights=[.2]*5,class_order=list(CLASSES),split_sha256=sha256(SPLIT)))
    summary=dict(status='completed',accuracy=m['accuracy'],macro_precision=m['macro_precision'],macro_recall=m['macro_recall'],macro_f1=m['macro_f1'],
        reference_accuracy=bm['accuracy'],reference_macro_f1=bm['macro_f1'],gained=gained,lost=lost,net=net,correct=int(new.sum()),incorrect=int((~new).sum()),
        melanoma_recall=m['per_class']['mel']['recall'],akiec_recall=m['per_class']['akiec']['recall'],
        reference_nll=float(-np.log(np.maximum(base[np.arange(len(y)),y],1e-12)).mean()),calibrated_nll=loss,
        reference_ece10=a,calibrated_ece10=b,material_gate_passed=bool(material),member_argmax_changes=0,
        positive_folds=sum(r['net']>0 for r in fold_rows),folds=fold_rows,temperature_range=[min(r['temperature'] for r in fit_rows),max(r['temperature'] for r in fit_rows)],
        test_loaded=False,gpu_used=False,deployment='No full-validation temperature fit; these are held-out-meta outputs',
        limitations=PLAN['scope']+'; cross-fitting protects temperature fitting only, not prior base checkpoint selection',
        decision='Material gate passed' if material else 'Material gate failed; preserve reference; no tuning or automatic GPU runs')
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'verification.json',dict(status='passed',fold_count=5,meta_holdout_coverage=1503,lesion_overlap=0,
        saved_metrics_independently_recomputed=True,member_argmax_changes=0,source_probabilities_unchanged=True,test_loaded=False))
    retry_registry_upsert(dict(experiment_id='s63_crossfit_temperature_calibration_exploratory_seed42',era='structured',record_kind='cpu_calibration_study',
        phase='post_test_exploratory_development',protocol='exploratory_image_level',evaluation_split='validation_meta_oof',split_manifest=relative(SPLIT),split_sha256=sha256(SPLIT),
        method='Fixed scalar temperature per member by train-fold NLL; five lesion-group meta folds; equal soft voting',ensemble_members=json.dumps([s['run'] for s in sources]),ensemble_weights=json.dumps([.2]*5),
        epochs=0,seed=42,status='completed',decision='material_gate_passed' if material else 'material_gate_failed',metrics_path=relative(OUT/'validation_meta_oof_metrics.json'),plots_dir=relative(OUT/'figures'),
        notes=summary['limitations']+'; no full-validation deployed temperature fit.',**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
