"""S82: one fixed equal-six addition, retaining both Tiny representations.

Saved exploratory validation probabilities only. No fitting, GPU or test scoring.
"""
import json
import time
import numpy as np
import pandas as pd
from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json, atomic_text
from research.final_cbam_development.runtime import require_launch_freeze
from research.final_cbam_development.protocol import verified_development
from research.final_cbam_development.artifacts import fp32_parts, COLS
from research.short_screening.lesion_bag_screen import report
from research.plots import metric_figures, comparison_figures, validate_metrics
from research.registry import upsert

OUT=ROOT/'results/short_screening/s82_cbam_addition_cpu'
RID='s82_cbam_addition_equal_six_exploratory_seed42'


def package(folder, val, p, title):
    m=report(val.label.to_numpy(),p);validate_metrics(m)
    f=val[['image_id','lesion_id']].copy();f['true_class']=val.diagnosis
    f['predicted_class']=[CLASSES[i] for i in p.argmax(1)];f[COLS]=p
    write_json(folder/'validation_metrics.json',m)
    write_csv(folder/'validation_predictions.csv',f.to_dict('records'))
    write_csv(folder/'validation_probabilities.csv',f[['image_id']+COLS].to_dict('records'))
    write_csv(folder/'per_class_metrics.csv',[dict(class_name=k,**v) for k,v in m['per_class'].items()])
    metric_figures(m,folder/'figures',title+' | exploratory validation')
    return m


def main():
    if (OUT/'summary.json').exists():
        print('Completed S82 preserved; no repeated experiment');return
    started=time.perf_counter();c,sig=require_launch_freeze();_,val=verified_development(c)
    source=ROOT/c['results']/'validation_predictions.csv';frame=pd.read_csv(source)
    assert frame.image_id.tolist()==val.image_id.tolist() and frame.true_class.tolist()==val.diagnosis.tolist()
    cbam=frame[COLS].to_numpy(dtype=np.float32)
    assert np.isfinite(cbam).all() and (cbam>=0).all() and np.allclose(cbam.sum(1),1,atol=1e-5)
    sources=sig['phase1']['ensemble']['sources']+[dict(run=c['experiment_id'],checkpoint=c['checkpoints']+'/best.pt',
        checkpoint_sha256=sha256(ROOT/c['checkpoints']/'best.pt'),prediction_file=relative(source),prediction_sha256=sha256(source))]
    plan=dict(date='2026-10-09',question='Does retaining Tiny and adding the new CBAM model recover complementary errors lost by replacement?',
        rule='Equal1/6 probability average of the exact S53 five checkpoints plus S79 accuracy-selected epoch33; no new fitting',
        sources=sources,protocol=c['protocol'],split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
        precision='saved FP32 identity probabilities',weights=[1/6]*6,
        material_gate=dict(minimum_accuracy_gain=.005,minimum_net_correct=8,macro_f1_nondecreasing=True,melanoma_recall_nondecreasing=True),
        search='Exactly one combination; no weighting,epoch/member search,calibration or alternative test model',
        test_loaded=False,gpu_used=False,training_epochs=0,
        caveat='Repeated development validation after known held-out outcomes; no independent test-performance claim; S80/S81 preserved')
    if (OUT/'PREDECLARED_PLAN.json').exists():assert json.loads((OUT/'PREDECLARED_PLAN.json').read_text())==plan
    else:write_json(OUT/'PREDECLARED_PLAN.json',plan)
    parts=fp32_parts(sig,val);base=np.mean(parts,axis=0);candidate=np.mean(parts+[cbam],axis=0)
    reference=report(val.label.to_numpy(),base)
    assert reference['accuracy']==.936127744510978 and abs(reference['macro_f1']-.8868574111567237)<1e-12
    m=package(OUT,val,candidate,'S82 fixed equal-six with Tiny and Tiny CBAM')
    old=base.argmax(1);new=candidate.argmax(1);y=val.label.to_numpy();s=cbam.argmax(1)
    gained=(old!=y)&(new==y);lost=(old==y)&(new!=y);gain=int(gained.sum());loss=int(lost.sum());net=gain-loss
    passed=net>=8 and m['accuracy']-reference['accuracy']>=.005-1e-12 and m['macro_f1']>=reference['macro_f1'] and m['per_class']['mel']['recall']>=reference['per_class']['mel']['recall']
    errors=[];classes=[]
    for image_id,truth,a,b,pc,g,l in zip(val.image_id,y,old,new,s,gained,lost):
        if g or l:errors.append(dict(image_id=image_id,true_class=CLASSES[truth],reference_class=CLASSES[a],candidate_class=CLASSES[b],cbam_class=CLASSES[pc],change='gained' if g else 'lost'))
    for i,name in enumerate(CLASSES):
        mask=y==i
        classes.append(dict(class_name=name,support=int(mask.sum()),reference_correct=int(((old==y)&mask).sum()),
            candidate_correct=int(((new==y)&mask).sum()),gained=int((gained&mask).sum()),lost=int((lost&mask).sum()),
            cbam_fixes_reference_errors=int(((old!=y)&(s==y)&mask).sum()),reference_fixes_cbam_errors=int(((old==y)&(s!=y)&mask).sum())))
    write_csv(OUT/'changed_predictions.csv',errors,['image_id','true_class','reference_class','candidate_class','cbam_class','change'])
    write_csv(OUT/'complementarity_by_class.csv',classes)
    comparison_figures([dict(display_name='S53 equal-five FP32 reference',accuracy=reference['accuracy'],macro_f1=reference['macro_f1']),
        dict(display_name='S82 equal-six CBAM addition',accuracy=m['accuracy'],macro_f1=m['macro_f1'])],OUT/'comparison_figures','Fixed addition instead of replacement | exploratory validation')
    summary=dict(status='completed',accuracy=m['accuracy'],macro_f1=m['macro_f1'],reference_accuracy=reference['accuracy'],reference_macro_f1=reference['macro_f1'],
        gained=gain,lost=loss,net_correct=net,accuracy_gain_pp=100*(m['accuracy']-reference['accuracy']),
        melanoma_recall=m['per_class']['mel']['recall'],reference_melanoma_recall=reference['per_class']['mel']['recall'],material_gate_passed=bool(passed),
        cbam_fixes_reference_errors=int(((old!=y)&(s==y)).sum()),reference_fixes_cbam_errors=int(((old==y)&(s!=y)).sum()),
        decision='promising_development_candidate' if passed else 'reject_no_material_gain',runtime_seconds=time.perf_counter()-started,
        test_loaded=False,gpu_used=False,source_checkpoint_hashes_unchanged=all(sha256(ROOT/a['checkpoint'])==a['checkpoint_sha256'] for a in sources))
    assert np.array_equal(report(y,candidate)['confusion_matrix'],m['confusion_matrix']) and summary['source_checkpoint_hashes_unchanged']
    write_json(OUT/'summary.json',summary);write_json(OUT/'verification.json',dict(status='passed',samples=1503,probabilities_finite_normalized=True,
        source_hashes_verified=True,metrics_recomputed=True,reference_fp32_reproduced=True,test_loaded=False,gpu_used=False))
    upsert(dict(experiment_id=RID,era='structured',record_kind='fixed_probability_fusion',phase='post_final_development',protocol=c['protocol'],
        evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],method=plan['rule'],
        model='convnext_tiny+convnext_small+densenet201+efficientnet_v2_s+efficientnet_b0+convnext_tiny_cbam',
        attention='CBAM in added S79 component',ensemble_members=json.dumps([a['run'] for a in sources]),ensemble_weights=json.dumps([1/6]*6),
        image_size=224,seed=42,epochs=0,status='completed',decision=summary['decision'],metrics_path=relative(OUT/'validation_metrics.json'),plots_dir=relative(OUT/'figures'),
        config_path=relative(OUT/'PREDECLARED_PLAN.json'),runtime_seconds=summary['runtime_seconds'],notes=plan['caveat'],
        **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    text=f"""# S82 fixed CBAM addition closeout

9October2026. Report writing was paused at the user's request to resume accuracy development. One CPU-only experiment retained the full S53 ensemble and added S79 epoch33 rather than replacing Tiny. Equal1/6 probabilities, unchanged saved FP32 identity predictions, same1503 exploratory validation images. No training, new inference, test access or search.

| Method | Accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|
| S53 retained equal-five | {100*reference['accuracy']:.4f}% | {reference['macro_f1']:.6f} | {100*reference['per_class']['mel']['recall']:.2f}% |
| S82 equal-six CBAM addition | {100*m['accuracy']:.4f}% | {m['macro_f1']:.6f} | {100*m['per_class']['mel']['recall']:.2f}% |

Gain{gain}, loss{loss}, net{net:+d} correct predictions. CBAM alone fixes{summary['cbam_fixes_reference_errors']} reference errors; the reference fixes{summary['reference_fixes_cbam_errors']} CBAM errors. The actual probability fusion, not an oracle union of correct answers, determines the result. Class-wise complementarity and changed identities are saved.

Material gate:>=8netcorrect and>=0.005absolute accuracy, macro-F1/melanoma recall nondecreasing. Passed:{passed}. Decision:{summary['decision']}. No test result is changed or forecast by this experiment. S80/S81 remain frozen historical evidence. Validation has been reused repeatedly and original test outcomes are already known; this is additional exploratory development, not independent confirmation.

Artifacts:`results/short_screening/s82_cbam_addition_cpu/`: plan/hashes, metrics, predictions/probabilities, class metrics, raw/normalized confusion and comparison PNG/PDF, complementarity and verification. Source checkpoints are unchanged.
"""
    atomic_text(ROOT/'research/short_screening/S82_CBAM_ADDITION_CLOSEOUT.md',text)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
