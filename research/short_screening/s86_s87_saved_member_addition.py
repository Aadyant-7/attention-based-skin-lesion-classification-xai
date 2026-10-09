"""Bounded saved-model expansion of S83, with an explicit development selection rule."""
import json
import time
import numpy as np
import pandas as pd
from research.common import ROOT, CLASSES, sha256, relative, write_json, write_csv, atomic_text
from research.final_cbam_development.runtime import require_launch_freeze
from research.final_cbam_development.protocol import verified_development
from research.final_cbam_development.artifacts import COLS
from research.short_screening.s82_cbam_addition_cpu import package
from research.short_screening.lesion_bag_screen import report
from research.plots import comparison_figures
from research.registry import upsert

OUT=ROOT/'results/short_screening/s86_s87_saved_member_addition'
CANDIDATES=[
    ('MobileNetV3-Large','s02_mobilenet_v3_large_none_exploratory_seed42','structured'),
    ('ResNet101','s19_resnet101_none_exploratory_seed42','structured'),
    ('ConvNeXt-Tiny ImageNet22k','s59_convnext_tiny_supervised22k_exploratory_seed42','22k'),
    ('PanDerm Large frozen SVM','s77_panderm_large_fp32_svm_exploratory_seed42','pan')]


def aligned(path,val):
    f=pd.read_csv(path)
    assert f.image_id.is_unique and set(f.image_id)==set(val.image_id)
    f=f.set_index('image_id').loc[val.image_id].reset_index()
    assert f.true_class.tolist()==val.diagnosis.tolist()
    p=f[COLS].to_numpy(dtype=np.float64)
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    return p


def main():
    if (OUT/'summary.json').exists(): print('Completed bounded study preserved');return
    start=time.perf_counter();c,_=require_launch_freeze();_,val=verified_development(c);y=val.label.to_numpy()
    reference=ROOT/'results/short_screening/s83_cbam_f1_addition_cpu/validation_predictions.csv'
    base=aligned(reference,val);bm=report(y,base);old=base.argmax(1);correct=old==y
    assert bm['accuracy']==.9401197604790419
    sources=[]
    for name,rid,kind in CANDIDATES:
        if kind=='structured':
            folder=ROOT/'results/structured_experiments'/rid;ck=ROOT/'checkpoints/structured'/rid
            pred=folder/'validation_predictions_macro_f1.csv';checkpoint=ck/'best_macro_f1.pt'
        elif kind=='22k':
            folder=ROOT/'results/short_screening/supervised22k_transfer_v1'/rid
            ck=ROOT/'checkpoints/short_screening/supervised22k_transfer_v1'/rid
            pred=folder/'validation_predictions_macro_f1.csv';checkpoint=ck/'best_macro_f1.pt'
        else:
            pred=ROOT/'results/short_screening/s77_panderm_large_transfer/large/validation_predictions.csv'
            checkpoint=ROOT/'checkpoints/short_screening/s77_panderm_large_transfer/large/svm.joblib'
        assert pred.is_file() and checkpoint.is_file()
        sources.append(dict(name=name,run=rid,prediction_file=relative(pred),prediction_sha256=sha256(pred),
            checkpoint=relative(checkpoint),checkpoint_sha256=sha256(checkpoint),
            selector='Standalone macro-F1 winner' if kind!='pan' else 'Existing fixed C10 weighted RBF SVM',
            precision='original saved AMP validation probabilities' if kind=='structured' else 'saved FP32 feature/inference probabilities'))
    original_sources=json.loads((ROOT/'results/short_screening/s83_cbam_f1_addition_cpu/PREDECLARED_PLAN.json').read_text())['sources']
    plan=dict(date='2026-10-09',question='Do saved unused backbones recover useful errors when added to the new equal-six CBAM ensemble?',
        original_six_sources=original_sources,
        reference_file=relative(reference),reference_sha256=sha256(reference),sources=sources,
        protocol=c['protocol'],split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
        screening='Four fixed saved candidates, one existing standalone selection per candidate; no refitting',
        selection_rule='Rank by number of S83 errors recovered, descending; ties by standalone macro-F1, then name. Require at least 15 recovered and >=0.85 standalone accuracy. Select at most two.',
        fusion_rule='Each selected candidate separately added to S83 at equal one-seventh. Retain all six existing sources. No weights or checkpoints searched.',
        conditional_joint='Only if both additions improve accuracy with macro-F1 and melanoma recall nondecreasing: one equal-eight addition of both. Otherwise no joint run.',
        material_gate=dict(minimum_net_correct=8,minimum_accuracy_gain=.005,macro_f1_nondecreasing=True,melanoma_recall_nondecreasing=True),
        caveat='Adaptive repeated validation selection; label-based screening is exploratory, not independent confirmation. Sources share the cohort, but recipes and original inference precision differ.',
        test_loaded=False,gpu_used=False,training_epochs=0,maximum_fusions=3)
    write_json(OUT/'PREDECLARED_PLAN.json',plan)
    audit=[];arrays={};class_rows=[]
    for source in sources:
        p=aligned(ROOT/source['prediction_file'],val);arrays[source['run']]=p;m=report(y,p);pred=p.argmax(1)
        fixes=(old!=y)&(pred==y);harms=(old==y)&(pred!=y)
        audit.append(dict(**source,accuracy=m['accuracy'],macro_f1=m['macro_f1'],recovers_s83_errors=int(fixes.sum()),
            errors_on_s83_correct=int(harms.sum()),melanoma_recall=m['per_class']['mel']['recall']))
        for i,cl in enumerate(CLASSES):
            mask=y==i;class_rows.append(dict(model=source['name'],class_name=cl,
                recovered=int((fixes&mask).sum()),errors_on_reference_correct=int((harms&mask).sum()),**m['per_class'][cl]))
    audit.sort(key=lambda a:(-a['recovers_s83_errors'],-a['macro_f1'],a['name']))
    selected=[a for a in audit if a['recovers_s83_errors']>=15 and a['accuracy']>=.85][:2]
    write_json(OUT/'candidate_audit.json',dict(candidates=audit,selected=[a['run'] for a in selected],
        warning='Recovered errors are a label-based diagnostic, not an oracle ensemble score.'))
    write_csv(OUT/'candidate_class_complementarity.csv',class_rows)
    rows=[dict(display_name='S83 equal-six reference',accuracy=bm['accuracy'],macro_f1=bm['macro_f1'])];results=[]
    jobs=[dict(id=86+i,name='add_'+a['run'].split('_')[0],members=[a]) for i,a in enumerate(selected)]
    for job in jobs:
        members=job['members'];p=(6*base+sum(arrays[a['run']] for a in members))/(6+len(members))
        path=OUT/job['name'];m=package(path,val,p,'S'+str(job['id'])+' saved-member addition')
        new=p.argmax(1);gainmask=(old!=y)&(new==y);lossmask=(old==y)&(new!=y)
        gain=int(gainmask.sum());loss=int(lossmask.sum());net=gain-loss
        directional=m['accuracy']>bm['accuracy'] and m['macro_f1']>=bm['macro_f1'] and m['per_class']['mel']['recall']>=bm['per_class']['mel']['recall']
        passed=directional and net>=8 and m['accuracy']-bm['accuracy']>=.005-1e-12
        result=dict(id=job['id'],name=job['name'],members=members,accuracy=m['accuracy'],macro_f1=m['macro_f1'],
            macro_precision=m['macro_precision'],macro_recall=m['macro_recall'],melanoma_recall=m['per_class']['mel']['recall'],
            gained=gain,lost=loss,net_correct=net,correct=int((new==y).sum()),incorrect=int((new!=y).sum()),
            directional_gate_passed=bool(directional),material_gate_passed=bool(passed),
            decision='material_gain_candidate' if passed else ('directional_only' if directional else 'reject_no_balanced_gain'))
        results.append(result);write_json(path/'summary.json',result)
        changes=[dict(image_id=val.iloc[i].image_id,true_class=CLASSES[y[i]],reference_class=CLASSES[old[i]],candidate_class=CLASSES[new[i]],change='gained' if gainmask[i] else 'lost')
            for i in np.flatnonzero(gainmask|lossmask)]
        write_csv(path/'changed_predictions.csv',changes,['image_id','true_class','reference_class','candidate_class','change'])
        write_json(path/'verification.json',dict(status='passed',samples=1503,source_probabilities_finite_normalized=True,cohort_alignment_verified=True,test_loaded=False,gpu_used=False))
        rows.append(dict(display_name=f"S{job['id']} S83 + "+' + '.join(a['name'] for a in members),accuracy=m['accuracy'],macro_f1=m['macro_f1']))
        upsert(dict(experiment_id=f"s{job['id']}_s83_{job['name']}_exploratory_seed42",era='structured',record_kind='fixed_probability_fusion',phase='post_final_development',
            protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],method='Equal averaging of unchanged S83 six sources plus saved additional member(s)',
            ensemble_members=json.dumps([a['run'] for a in original_sources]+[a['run'] for a in members]),ensemble_weights=json.dumps([1/(6+len(members))]*(6+len(members))),
            attention='CBAM in S79 source only',image_size=224,seed=42,epochs=0,status='completed',decision=result['decision'],
            config_path=relative(OUT/'PREDECLARED_PLAN.json'),metrics_path=relative(path/'validation_metrics.json'),plots_dir=relative(path/'figures'),
            confusion_matrix_path=relative(path/'figures/confusion_matrix.csv'),val_loss=m['loss'],notes=plan['caveat'],
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
        if len(results)==2 and all(r['directional_gate_passed'] for r in results):
            jobs.append(dict(id=88,name='joint_equal_eight',members=selected))
    comparison_figures(rows,OUT/'comparison_figures','Bounded saved-model expansion | exploratory validation')
    assert sha256(reference)==plan['reference_sha256']
    assert all(sha256(ROOT/a['prediction_file'])==a['prediction_sha256'] and sha256(ROOT/a['checkpoint'])==a['checkpoint_sha256'] for a in sources)
    summary=dict(status='completed',reference_accuracy=bm['accuracy'],reference_macro_f1=bm['macro_f1'],audit=audit,results=results,
        source_hashes_unchanged=True,test_loaded=False,gpu_used=False,runtime_seconds=time.perf_counter()-start,
        joint_run_performed=len(results)==3,highest_observed_accuracy=max([bm['accuracy']]+[r['accuracy'] for r in results]))
    write_json(OUT/'summary.json',summary)
    text='# S86/S87 bounded saved-model additions\n\nFour fixed unused candidates were screened for error complementarity; at most two separately added and a joint run allowed only if both passed the directional gate. This is adaptive exploratory validation selection, not independent confirmation. No training, fresh inference or test access.\n\n'
    text+='| Method | Accuracy | Macro-F1 | Gained / lost vs S83 |\n|---|---:|---:|---:|\n'
    text+=f"| S83 reference | {100*bm['accuracy']:.4f}% | {bm['macro_f1']:.6f} | — |\n"
    for r in results:text+=f"| S{r['id']} {r['name']} | {100*r['accuracy']:.4f}% | {r['macro_f1']:.6f} | {r['gained']} / {r['lost']} |\n"
    text+='\nAdding models is conditional on measured probability-fusion performance, not the union of all correct predictions. All source hashes, selection rules, precision differences, candidate/class diagnostics, changed identities, scores and PNG/PDF figures are saved. No weights, checkpoints or temperatures were searched.\n'
    atomic_text(ROOT/'research/short_screening/S86_S87_SAVED_ADDITION_CLOSEOUT.md',text)
    print(json.dumps(dict(status='completed',audit=[{k:a[k] for k in ['name','accuracy','macro_f1','recovers_s83_errors','errors_on_s83_correct']} for a in audit],results=results,joint_run_performed=summary['joint_run_performed']),indent=2))


if __name__=='__main__':main()
