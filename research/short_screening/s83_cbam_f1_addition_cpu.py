"""One saved accuracy-vs-macro-F1 selection-rule comparison for CBAM addition."""
import json
import time
import numpy as np
import pandas as pd
from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json, atomic_text
from research.final_cbam_development.runtime import require_launch_freeze
from research.final_cbam_development.protocol import verified_development
from research.final_cbam_development.artifacts import fp32_parts, COLS
from research.short_screening.lesion_bag_screen import report
from research.short_screening.s82_cbam_addition_cpu import package
from research.plots import comparison_figures
from research.registry import upsert

OUT=ROOT/'results/short_screening/s83_cbam_f1_addition_cpu'
RID='s83_cbam_f1_addition_equal_six_exploratory_seed42'


def main():
    if (OUT/'summary.json').exists():print('Completed S83 preserved');return
    start=time.perf_counter();c,sig=require_launch_freeze();_,val=verified_development(c)
    source=ROOT/c['results']/'macro_f1_selected/validation_predictions.csv';checkpoint=ROOT/c['checkpoints']/'best_macro_f1.pt'
    frame=pd.read_csv(source);assert frame.image_id.tolist()==val.image_id.tolist() and frame.true_class.tolist()==val.diagnosis.tolist()
    p=frame[COLS].to_numpy(dtype=np.float32);assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    summary79=json.loads((ROOT/c['results']/'training_summary.json').read_text());assert summary79['best_macro_f1_epoch']==35
    sources=sig['phase1']['ensemble']['sources']+[dict(run=c['experiment_id'],selector='earliest standalone macro-F1 maximum',epoch=35,
        checkpoint=relative(checkpoint),checkpoint_sha256=sha256(checkpoint),prediction_file=relative(source),prediction_sha256=sha256(source))]
    plan=dict(date='2026-10-09',question='Does the already-saved macro-F1-selected CBAM checkpoint improve equal-six class balance versus its accuracy-selected counterpart?',
        rule='Exact S53 five-model FP32 probabilities plus S79 standalone macro-F1 maximum at35; equal1/6; no further checkpoint trials',
        selection_ablation='Two previously saved standalone rules only: S82 accuracy33 versus S83 macro-F1 35; no scanning other epochs',
        sources=sources,weights=[1/6]*6,protocol=c['protocol'],split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
        material_gate=dict(minimum_net_correct=8,minimum_accuracy_gain=.005,macro_f1_nondecreasing=True,melanoma_recall_nondecreasing=True),
        test_loaded=False,gpu_used=False,training_epochs=0,caveat='Repeated exploratory validation after known test outcomes; not independent test evidence')
    if (OUT/'PREDECLARED_PLAN.json').exists():assert json.loads((OUT/'PREDECLARED_PLAN.json').read_text())==plan
    else:write_json(OUT/'PREDECLARED_PLAN.json',plan)
    parts=fp32_parts(sig,val);base=np.mean(parts,axis=0);combined=np.mean(parts+[p],axis=0)
    ref=report(val.label.to_numpy(),base);assert ref['accuracy']==.936127744510978
    m=package(OUT,val,combined,'S83 equal-six with macro-F1-selected CBAM')
    prev=json.loads((ROOT/'results/short_screening/s82_cbam_addition_cpu/validation_metrics.json').read_text())
    y=val.label.to_numpy();old=base.argmax(1);new=combined.argmax(1);gainmask=(old!=y)&(new==y);lossmask=(old==y)&(new!=y)
    gain=int(gainmask.sum());loss=int(lossmask.sum());net=gain-loss
    passed=net>=8 and m['accuracy']-ref['accuracy']>=.005-1e-12 and m['macro_f1']>=ref['macro_f1'] and m['per_class']['mel']['recall']>=ref['per_class']['mel']['recall']
    rows=[];classrows=[]
    for i,row in val.iterrows():
        if gainmask[i] or lossmask[i]:rows.append(dict(image_id=row.image_id,true_class=row.diagnosis,reference_class=CLASSES[old[i]],candidate_class=CLASSES[new[i]],change='gained' if gainmask[i] else 'lost'))
    for i,name in enumerate(CLASSES):
        mask=y==i;classrows.append(dict(class_name=name,support=int(mask.sum()),reference_correct=int(((old==y)&mask).sum()),
            candidate_correct=int(((new==y)&mask).sum()),gained=int((gainmask&mask).sum()),lost=int((lossmask&mask).sum())))
    write_csv(OUT/'changed_predictions.csv',rows,['image_id','true_class','reference_class','candidate_class','change']);write_csv(OUT/'class_comparison.csv',classrows)
    comparison_figures([dict(display_name='S53 equal-five reference',accuracy=ref['accuracy'],macro_f1=ref['macro_f1']),
        dict(display_name='S82 added accuracy-selected CBAM',accuracy=prev['accuracy'],macro_f1=prev['macro_f1']),
        dict(display_name='S83 added macro-F1-selected CBAM',accuracy=m['accuracy'],macro_f1=m['macro_f1'])],OUT/'comparison_figures','Fixed CBAM addition selection-rule ablation | exploratory validation')
    result=dict(status='completed',accuracy=m['accuracy'],macro_f1=m['macro_f1'],macro_precision=m['macro_precision'],macro_recall=m['macro_recall'],
        gained=gain,lost=loss,net_correct=net,accuracy_gain_pp=100*(m['accuracy']-ref['accuracy']),melanoma_recall=m['per_class']['mel']['recall'],
        reference_melanoma_recall=ref['per_class']['mel']['recall'],s82_accuracy=prev['accuracy'],s82_macro_f1=prev['macro_f1'],
        material_gate_passed=bool(passed),decision='promising_development_candidate' if passed else 'reject_no_material_gain',
        runtime_seconds=time.perf_counter()-start,test_loaded=False,gpu_used=False)
    assert all(sha256(ROOT/a['checkpoint'])==a['checkpoint_sha256'] for a in sources)
    write_json(OUT/'summary.json',result);write_json(OUT/'verification.json',dict(status='passed',samples=1503,source_hashes_unchanged=True,
        metrics_recomputed=True,reference_reproduced=True,probabilities_finite_normalized=True,test_loaded=False,gpu_used=False))
    upsert(dict(experiment_id=RID,era='structured',record_kind='fixed_probability_fusion',phase='post_final_development',protocol=c['protocol'],evaluation_split='validation',
        split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],model='convnext_tiny+convnext_small+densenet201+efficientnet_v2_s+efficientnet_b0+convnext_tiny_cbam',
        method=plan['rule'],attention='CBAM in added S79 macro-F1-selected component',ensemble_members=json.dumps([a['run'] for a in sources]),ensemble_weights=json.dumps([1/6]*6),
        image_size=224,seed=42,epochs=0,status='completed',decision=result['decision'],config_path=relative(OUT/'PREDECLARED_PLAN.json'),metrics_path=relative(OUT/'validation_metrics.json'),
        plots_dir=relative(OUT/'figures'),runtime_seconds=result['runtime_seconds'],notes=plan['caveat'],**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    text=f"""# S83 fixed macro-F1 CBAM addition closeout

One CPU-only comparison uses the already-saved standalone macro-F1-selected S79 checkpoint35 rather than the accuracy-selected33 used by S82. All five S53 sources and equal1/6 fusion are unchanged. These are the two recorded standalone selection rules, not a sweep over training epochs. No training, new inference or test access.

| Method | Exploratory accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|
| S53 retained equal-five | {100*ref['accuracy']:.4f}% | {ref['macro_f1']:.6f} | {100*ref['per_class']['mel']['recall']:.2f}% |
| S82 added accuracy-selected CBAM | {100*prev['accuracy']:.4f}% | {prev['macro_f1']:.6f} | {100*prev['per_class']['mel']['recall']:.2f}% |
| S83 added macro-F1-selected CBAM | {100*m['accuracy']:.4f}% | {m['macro_f1']:.6f} | {100*m['per_class']['mel']['recall']:.2f}% |

Gained{gain}, lost{loss}, net{net:+d} correct versus S53. Material gate passed:{passed}; decision:{result['decision']}. All metrics/predictions/probabilities, class/confusion/comparison PNG/PDF, changed identities, source hashes and verification are saved in `results/short_screening/s83_cbam_f1_addition_cpu/`.

Repeated-validation selection and post-test development remain limitations. S80/S81 and their test/XAI results stay unchanged; neither new six-model method has been evaluated on a new independent test. No further checkpoint/weight search or GPU run is queued.
"""
    atomic_text(ROOT/'research/short_screening/S83_CBAM_F1_ADDITION_CLOSEOUT.md',text);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
