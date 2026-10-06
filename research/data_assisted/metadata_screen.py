"""Fixed train-only clinical metadata likelihood adjustment of existing S53."""
import json,time
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from research.common import ROOT,CLASSES,sha256,relative,write_csv,write_json
from research.strict_train import metric_report,run_lock
from research.short_screening.convnextv2_transfer import config,development
from research.aggressive.core import predictions,retry_registry_upsert
from research.plots import metric_figures,comparison_figures

OUT=ROOT/'results/data_assisted/s68_s69_metadata_screen'
REF=ROOT/'results/short_screening/s53_equal_five_b0_addition'
META=ROOT/'data/raw/HAM10000/HAM10000_metadata.csv'
FIELDS=['age_bin','sex','localization']
VARIANTS={'s68_age':['age_bin'],'s69_age_sex_location':FIELDS}
PLAN=dict(question='Does train-only clinically available metadata help the retained S53 ensemble?',
    reference='Existing S53 equal-five probabilities; all checkpoints/weights/preprocessing remain unchanged',
    candidates=VARIANTS,formula='softmax(log(P_S53) +0.25*sum_f log(P(metadata_field_f|class)))',
    strength=.25,laplace_alpha=1.,age_bins='[0,20),[20,40),[40,60),[60,80),[80,120]; no learned boundaries',
    fit='HAM training only; exclude every training lesion present in validation; one categorical metadata record per lesion',
    conflict_policy='Exclude a fit lesion with contradictory observed categorical field values or classes; missing fields are neutral at inference',
    unknown_policy='Missing age/sex/location and unseen training-vocabulary categories add zero log adjustment; missingness is not a predictor',
    prohibited='No image/lesion ID, diagnosis confirmation, diagnosis-derived attributes or test metadata as predictors',
    material_gate='At least8 net additional correct images AND accuracy gain>=.005 AND macroF1 and melanoma recall nondecreasing',
    limit='Exactly two predefined field ablations, one strength and smoothing; no alternative bins/strengths/thresholds/source search',
    caveat='Conditional independence is an approximation; image features already encode some clinical correlates. Existing base checkpoints were selected on this repeatedly used exploratory cohort; not independent test evidence.',
    source='https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes',test_loaded=False,gpu_used=False)


def encoded(frame):
    out=frame.copy();age=pd.to_numeric(out.age,errors='coerce')
    assert not ((age.dropna()<0)|(age.dropna()>120)).any()
    out['age_bin']=[str(int(np.searchsorted([20,40,60,80],a,side='right'))) if pd.notna(a) else None for a in age]
    for field in ['sex','localization']:
        out[field]=out[field].map(lambda x:None if pd.isna(x) or str(x).strip().lower() in ['unknown','nan','none',''] else str(x).strip().lower())
    return out


def save_result(name,ids,y,p,title):
    assert p.shape==(len(y),7) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-12)
    m=metric_report(y,p,float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))
    target=OUT/name;write_json(target/'validation_metrics.json',m)
    rows=predictions(ids,y,p);write_csv(target/'validation_predictions.csv',rows)
    write_csv(target/'validation_probabilities.csv',pd.DataFrame(rows)[['image_id']+[f'p_{cl}' for cl in CLASSES]].to_dict('records'))
    metric_figures(m,target/'figures',title)
    saved=pd.read_csv(target/'validation_predictions.csv');assert saved.image_id.tolist()==list(ids)
    check=metric_report(y,saved[[f'p_{cl}' for cl in CLASSES]].to_numpy(),m['loss'])
    assert check['confusion_matrix']==m['confusion_matrix'] and abs(check['macro_f1']-m['macro_f1'])<1e-12
    return m


def main():
    if (OUT/'summary.json').exists():print((OUT/'summary.json').read_text());return
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREDECLARED_PLAN.json').exists():assert json.loads((OUT/'PREDECLARED_PLAN.json').read_text())==PLAN
    else:write_json(OUT/'PREDECLARED_PLAN.json',PLAN)
    tick=time.perf_counter();c=config();train,val,_=development(c)
    allowed=set(pd.concat([train,val]).image_id)
    # Read all identity strings only; skip other rows BEFORE parsing clinical columns.
    identity=pd.read_csv(META,usecols=['image_id'])
    skip=set((identity.index[~identity.image_id.isin(allowed)]+1).tolist())
    meta=pd.read_csv(META,usecols=['image_id','age','sex','localization'],skiprows=lambda line:line in skip)
    assert set(meta.image_id)==allowed and meta.image_id.is_unique and len(meta)==8512
    ref=pd.read_csv(REF/'validation_predictions.csv');assert ref.image_id.is_unique and len(ref)==1503
    ids=ref.image_id.to_numpy();val=val.set_index('image_id').loc[ids].reset_index()
    y=val.label.to_numpy();assert ref.true_class.tolist()==[CLASSES[i] for i in y]
    base=ref[[f'p_{cl}' for cl in CLASSES]].to_numpy()
    expected=json.loads((REF/'validation_metrics.json').read_text())
    bm=save_result('s53_reference',ids,y,base,'S53 retained ensemble | matched metadata reference')
    assert bm['confusion_matrix']==expected['confusion_matrix'] and abs(bm['macro_f1']-expected['macro_f1'])<1e-12
    valmeta=encoded(val.merge(meta,on='image_id',validate='one_to_one'))
    fitmeta=encoded(train.merge(meta,on='image_id',validate='one_to_one'))
    overlap=fitmeta.lesion_id.isin(set(val.lesion_id));shared=int(overlap.sum())
    excluded=fitmeta.loc[overlap,['image_id','lesion_id']].copy();excluded['reason']='lesion_present_in_validation'
    fitmeta=fitmeta.loc[~overlap]
    records=[];conflicts=[]
    for lesion,group in fitmeta.groupby('lesion_id',sort=True):
        label=group.label.unique();values={f:group[f].dropna().unique().tolist() for f in FIELDS}
        if len(label)!=1 or any(len(v)>1 for v in values.values()):
            conflicts.extend(dict(image_id=r.image_id,lesion_id=lesion,reason='contradictory_observed_categorical_metadata') for r in group.itertuples(index=False));continue
        records.append(dict(image_id=min(group.image_id),lesion_id=lesion,label=int(label[0]),
            **{f:v[0] if v else None for f,v in values.items()}))
    fit=pd.DataFrame(records);assert fit.lesion_id.is_unique and not set(fit.lesion_id)&set(val.lesion_id)
    counts=np.bincount(fit.label,minlength=7);assert (counts>0).all()
    write_csv(OUT/'metadata_fit_lesions.csv',fit.to_dict('records'))
    er=excluded.to_dict('records')+conflicts
    write_csv(OUT/'excluded_metadata_fit_images.csv',er or [dict(image_id='',lesion_id='',reason='none')])
    tables={};logtables={};observed={};table_rows=[]
    for field in FIELDS:
        vocab=[str(i) for i in range(5)] if field=='age_bin' else sorted(fit[field].dropna().unique())
        assert len(vocab)>0
        field_counts=np.zeros((7,len(vocab)),dtype=int);known=np.zeros(7,dtype=int)
        for i in range(7):
            values=fit.loc[fit.label==i,field].dropna();known[i]=len(values)
            field_counts[i]=[int((values==v).sum()) for v in vocab]
        probs=(field_counts+1)/(known[:,None]+len(vocab));assert np.allclose(probs.sum(1),1)
        tables[field]=dict(vocabulary=vocab,known_fit_lesions_by_class=known.tolist(),counts=field_counts.tolist(),likelihoods=probs.tolist())
        logtables[field]={v:np.log(probs[:,j]) for j,v in enumerate(vocab)}
        observed[field]=int(valmeta[field].isin(vocab).sum())
        for i,cl in enumerate(CLASSES):
            for j,value in enumerate(vocab):table_rows.append(dict(field=field,category=value,class_name=cl,fit_count=int(field_counts[i,j]),observed_class_lesions=int(known[i]),likelihood=float(probs[i,j])))
    write_json(OUT/'fitted_metadata_model.json',dict(strength=.25,laplace_alpha=1.,age_boundaries=[20,40,60,80],class_order=list(CLASSES),
        fields=tables,fit_lesions=len(fit),fit_images_before_grouping=len(fitmeta),class_counts=counts.tolist(),train_validation_lesion_overlap=0))
    write_csv(OUT/'metadata_likelihoods.csv',table_rows)
    metrics={};comparisons=[];class_rows=[];changed=[]
    ac=base.argmax(1)==y
    for name,fields in VARIANTS.items():
        adjustment=np.zeros_like(base)
        for field in fields:
            for row,value in enumerate(valmeta[field]):
                if value in logtables[field]:adjustment[row]+=logtables[field][value]
        logits=np.log(np.maximum(base,1e-12))+.25*adjustment
        p=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
        m=save_result(name,ids,y,p,f'{name.upper()} train-only metadata | exploratory validation')
        bc=p.argmax(1)==y;gained=int((~ac&bc).sum());lost=int((ac&~bc).sum());net=int(bc.sum()-ac.sum())
        passed=net>=8 and m['accuracy']>=bm['accuracy']+.005-1e-12 and m['macro_f1']>=bm['macro_f1']-1e-12 and m['per_class']['mel']['recall']>=bm['per_class']['mel']['recall']-1e-12
        comparisons.append(dict(method=name,gained=gained,lost=lost,net_correct_change=net,material_gate_passed=bool(passed),
            accuracy_delta=m['accuracy']-bm['accuracy'],macro_f1_delta=m['macro_f1']-bm['macro_f1'],melanoma_recall=m['per_class']['mel']['recall']))
        for i,cl in enumerate(CLASSES):
            subset=y==i
            class_rows.append(dict(method=name,class_name=cl,**{f'reference_{k}':v for k,v in bm['per_class'][cl].items()},
                **{f'candidate_{k}':v for k,v in m['per_class'][cl].items()},gained=int((~ac&bc&subset).sum()),lost=int((ac&~bc&subset).sum())))
        for row in np.flatnonzero(base.argmax(1)!=p.argmax(1)):
            changed.append(dict(method=name,image_id=ids[row],true_class=CLASSES[y[row]],reference_class=CLASSES[base[row].argmax()],
                candidate_class=CLASSES[p[row].argmax()],reference_correct=bool(ac[row]),candidate_correct=bool(bc[row]),
                age_bin=valmeta.iloc[row].age_bin,sex=valmeta.iloc[row].sex,localization=valmeta.iloc[row].localization))
        metrics[name]=m
        retry_registry_upsert(dict(experiment_id=name+'_metadata_likelihood_exploratory_seed42',era='structured',record_kind='cpu_metadata_fusion',
            phase='post_test_exploratory_development',protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
            model='S53 equal-five ensemble',method='Fixed0.25 train-only categorical metadata likelihood adjustment; '+','.join(fields),
            epochs=0,seed=42,status='completed',decision='material_gate_passed' if passed else 'material_gate_failed',
            metrics_path=relative(OUT/name/'validation_metrics.json'),plots_dir=relative(OUT/name/'figures'),
            notes='Metadata fit on training-only lesions excluding any validation lesion; no validation fit/tuning. Reused exploratory CNNs previously selected on validation; no independent test claim.',
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    write_csv(OUT/'method_comparison.csv',comparisons);write_csv(OUT/'class_comparison.csv',class_rows)
    write_csv(OUT/'changed_predictions.csv',changed or [dict(method='',image_id='',true_class='',reference_class='',candidate_class='',reference_correct='',candidate_correct='',age_bin='',sex='',localization='')])
    comparison_figures([dict(display_name='S53 image-only',accuracy=bm['accuracy'],macro_f1=bm['macro_f1'])]+
        [dict(display_name=n,accuracy=m['accuracy'],macro_f1=m['macro_f1']) for n,m in metrics.items()],OUT/'comparison_figures','Fixed train-only metadata adjustment of retained ensemble')
    source_files=[META,REF/'validation_predictions.csv',REF/'validation_metrics.json',ROOT/c['split_manifest'],ROOT/'research/data_assisted/metadata_screen.py']
    write_json(OUT/'source_manifest.json',dict(files={relative(p):sha256(p) for p in source_files},class_order=list(CLASSES),
        clinical_columns_parsed=['age','sex','localization'],excluded_predictors=['image_id','lesion_id','dx','dx_type'],original_test_clinical_rows_parsed=False))
    write_json(OUT/'verification.json',dict(status='passed',reference_reproduced=True,validation_images=len(val),clinical_development_rows=len(meta),
        metadata_fit_lesions=len(fit),metadata_train_validation_lesion_overlap=0,excluded_shared_lesion_train_images=shared,
        contradictory_fit_images=len(conflicts),training_only_vocabularies_and_counts=True,validation_labels_used_for_fit=False,
        unknown_and_missing_fields_neutral=True,saved_metrics_and_confusion_recomputed=True,test_loaded=False,gpu_used=False))
    result=dict(status='completed',reference=bm,metrics=metrics,comparisons=comparisons,fit_lesions=len(fit),
        fit_class_counts=counts.tolist(),validation_known_fields=observed,excluded_shared_lesion_train_images=shared,contradictory_fit_images=len(conflicts),
        runtime_seconds=time.perf_counter()-tick,test_loaded=False,gpu_used=False,
        decision='Prepare separate confirmation only; no automatic deployment' if any(r['material_gate_passed'] for r in comparisons) else 'Keep S53; metadata gate failed; no strength/bin/search retries',
        limitations=PLAN['caveat'])
    write_json(OUT/'summary.json',result);print(json.dumps(dict(comparisons=comparisons,fit_lesions=len(fit),decision=result['decision']),indent=2))


if __name__=='__main__':
    with run_lock('s68_s69_metadata_screen'):main()
