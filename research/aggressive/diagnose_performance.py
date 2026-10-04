"""Read-only result audit: no training, no raw images, no new test evaluation."""
import json,math
import numpy as np
import pandas as pd
from research.common import ROOT,CLASSES,write_json,write_csv

OUT=ROOT/'results/aggressive_enhanced/v1/root_cause_audit'

def load(name):return json.loads((ROOT/name).read_text())

def main():
    exploratory=load('results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42/validation_metrics_identity_fp32.json')
    strict=load('results/final_strict/v1/ensemble/validation_metrics.json')
    test=load('results/final_locked_test/v1/ensemble/test_metrics.json')
    b3=load('results/aggressive_enhanced/v1/s32_efficientnet_b3_cbam_enhanced_seed42/validation_metrics.json')
    dense=load('results/aggressive_enhanced/v1/s33_densenet201_enhanced_seed42/validation_metrics.json')
    stages=[('exploratory_validation',exploratory),('strict_validation',strict),('original_test_saved_summary',test),('enhanced_b3_validation',b3),('incomplete_dense_validation',dense)]
    rows=[]
    for name,m in stages:
        cm=np.array(m['confusion_matrix']);n=int(cm.sum());incorrect=int(n-np.trace(cm))
        for i,cl in enumerate(CLASSES):
            p=m['per_class'][cl]
            rows.append(dict(result=name,class_name=cl,support=p['support'],correct=int(cm[i,i]),errors=int(cm[i].sum()-cm[i,i]),
                error_fraction_of_all_errors=float((cm[i].sum()-cm[i,i])/incorrect),precision=p['precision'],recall=p['recall'],f1=p['f1']))
    write_csv(OUT/'class_error_breakdown.csv',rows)
    cm=np.array(test['confusion_matrix']);pairs=[]
    for i,a in enumerate(CLASSES):
        for j,b in enumerate(CLASSES):
            if i!=j and cm[i,j]:pairs.append(dict(true_class=a,predicted_class=b,errors=int(cm[i,j])))
    write_csv(OUT/'original_test_confusions_from_saved_matrix.csv',sorted(pairs,key=lambda r:-r['errors']))
    # Read partition identities only, never test diagnosis/raw images.
    ids=pd.read_csv(ROOT/'data/splits/exploratory/image_level_dev_v1.csv',usecols=['image_id','lesion_id','split'])
    train_lesions=set(ids.loc[ids.split=='train','lesion_id']);val=ids.loc[ids.split=='val'].copy()
    val['training_lesion_seen']=val.lesion_id.isin(train_lesions)
    pred=pd.read_csv(ROOT/'results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42/validation_predictions_identity_fp32.csv')
    merged=val.merge(pred,on='image_id',validate='one_to_one');assert len(merged)==len(val)==1503
    merged['correct']=merged.true_class==merged.predicted_class
    cohorts=[]
    for seen,group in merged.groupby('training_lesion_seen'):
        cohorts.append(dict(training_lesion_seen=bool(seen),images=len(group),accuracy=float(group.correct.mean()),correct=int(group.correct.sum())))
    write_csv(OUT/'exploratory_seen_lesion_summary.csv',cohorts)
    subgroup=[]
    for (seen,cl),group in merged.groupby(['training_lesion_seen','true_class']):
        subgroup.append(dict(training_lesion_seen=bool(seen),class_name=cl,images=len(group),recall=float(group.correct.mean())))
    write_csv(OUT/'exploratory_seen_lesion_class_recall.csv',subgroup)
    class_mix=pred.true_class.value_counts(normalize=True).to_dict()
    standardized=[]
    for seen,g in pd.DataFrame(subgroup).groupby('training_lesion_seen'):
        standardized.append(dict(training_lesion_seen=bool(seen),class_standardized_accuracy=sum(class_mix[r.class_name]*r.recall for r in g.itertuples()),
            interpretation='Diagnostic reweighting to common exploratory validation class mix; not causal, not a test result'))
    write_json(OUT/'exploratory_class_standardized_comparison.json',standardized)
    curves=[];runtime=[]
    specs=[('proven_strict_convnext','results/structured_experiments/s29_convnext_tiny_final_strict_seed42/history.csv'),
           ('historical_exploratory_dense','results/structured_experiments/s15_densenet201_none_exploratory_seed42/history.csv'),
           ('enhanced_b3','results/aggressive_enhanced/v1/s32_efficientnet_b3_cbam_enhanced_seed42/history.csv'),
           ('incomplete_enhanced_dense','results/aggressive_enhanced/v1/s33_densenet201_enhanced_seed42/history.csv')]
    for name,path in specs:
        h=pd.read_csv(ROOT/path);full=h.loc[h.get('stage',pd.Series('full',index=h.index))=='full']
        runtime.append(dict(run=name,completed_epochs=len(h),total_epoch_minutes=float(h.epoch_seconds.sum()/60),
            full_epoch_seconds_median=float(full.epoch_seconds.median()),full_epoch_seconds_min=float(full.epoch_seconds.min()),
            full_epoch_seconds_max=float(full.epoch_seconds.max()),peak_allocated_mb=float(h.peak_allocated_vram_mb.max())))
        for epoch in [1,5,10,15,19,20,25,30,40]:
            through=h.loc[h.epoch<=epoch]
            if len(through)<epoch:continue
            last=through.iloc[-1]
            curves.append(dict(run=name,through_epoch=epoch,best_accuracy=float(through.val_accuracy.max()),best_macro_f1=float(through.val_macro_f1.max()),
                latest_accuracy=float(last.val_accuracy),latest_macro_f1=float(last.val_macro_f1)))
    write_csv(OUT/'matched_stage_learning_summary.csv',curves);write_csv(OUT/'runtime_summary.csv',runtime)
    counts=load('results/aggressive_enhanced/v1/class_distribution.json')['training_counts']
    n=sum(counts.values());base=np.sqrt(n/(7*np.array([counts[cl] for cl in CLASSES])));base/=base.mean()
    aggressive=load('results/aggressive_enhanced/v1/class_distribution.json')['class_weights']
    write_csv(OUT/'loss_weight_comparison.csv',[dict(class_name=cl,training_count=counts[cl],proven_sqrt_weight=float(base[i]),enhanced_weight=aggressive[cl],
        proven_ratio_to_nv=float(base[i]/base[5]),enhanced_ratio_to_nv=aggressive[cl]/aggressive['nv']) for i,cl in enumerate(CLASSES)])
    perclass_drop=[]
    val_total=int(np.array(strict['confusion_matrix']).sum());test_total=int(cm.sum())
    for cl in CLASSES:
        v=strict['per_class'][cl];t=test['per_class'][cl]
        contribution=v['support']/val_total*v['recall']-t['support']/test_total*t['recall']
        perclass_drop.append(dict(class_name=cl,validation_recall=v['recall'],test_recall=t['recall'],accuracy_drop_contribution_pp=100*contribution))
    write_csv(OUT/'descriptive_validation_test_gap.csv',perclass_drop)
    needed=math.ceil(.92*test_total)-int(np.trace(cm))
    summary=dict(scope='Saved metrics/predictions only; no training, raw image loading or new test inference',
        exploratory_accuracy=exploratory['accuracy'],strict_validation_accuracy=strict['accuracy'],original_test_accuracy=test['accuracy'],
        exploratory_to_strict_drop_pp=100*(exploratory['accuracy']-strict['accuracy']),strict_validation_to_test_drop_pp=100*(strict['accuracy']-test['accuracy']),
        original_test_errors=int(cm.sum()-np.trace(cm)),correct_predictions_needed_for_92_percent=needed,
        exploratory_seen_lesion_cohorts=cohorts,runtime=runtime,
        limitations=['Seen/unseen subgroups differ in class mix and lesion difficulty; not a causal leakage estimate.',
        'Validation-selected models and many development comparisons can create optimistic validation estimates.',
        'Aggressive package changes are confounded; no single causal attribution without ablations.',
        'DenseNet stopped at19; its eventual convergence is unknown.'],
        active_gpu_runs=0)
    write_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
