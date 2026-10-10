"""Verify S95 and execute its one predeclared S83 member replacement on CPU."""
import io
import json
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from research.common import ROOT, CLASSES, relative, sha256, write_json, write_csv, atomic_text
from research.final_cbam_development.runtime import require_launch_freeze
from research.final_cbam_development.protocol import verified_development
from research.final_cbam_development.artifacts import fp32_parts
from research.short_screening import s95_convnext_regularization as candidate
from research.short_screening.s92_s94_transformer_addition_cpu import verify_run
from research.short_screening.s86_s87_saved_member_addition import aligned
from research.short_screening.s82_cbam_addition_cpu import package
from research.short_screening.lesion_bag_screen import report
from research.plots import comparison_figures
from research.registry import read_registry, upsert

OUT=ROOT/'results/short_screening/s96_convnext_replacement_cpu'
RID='s96_s83_s95_replacement_equal_six_exploratory_seed42'
REFERENCE=ROOT/'results/short_screening/s83_cbam_f1_addition_cpu'
OLD='s06_convnext_tiny_none_exploratory_seed42'


def main():
    if (OUT/'summary.json').exists():
        print('Completed S96 preserved; no repeated scoring');return
    started=time.perf_counter();torch.set_num_threads(4)
    committed=pd.read_csv(io.StringIO(subprocess.check_output(['git','show','HEAD:results/master_experiment_registry.csv'],cwd=ROOT,text=True)),dtype=str).fillna('').to_dict('records')
    current={r['experiment_id']:r for r in read_registry()}
    assert all(r==current[r['experiment_id']] for r in committed if r['experiment_id']!=candidate.RID)
    c,sig=require_launch_freeze();_,val=verified_development(c);y=val.label.to_numpy()
    verification=verify_run('S95 ConvNeXt-Tiny stronger weight decay',candidate,val)
    selected=verification['selections'][1]
    original=json.loads((REFERENCE/'PREDECLARED_PLAN.json').read_text())
    sources=original['sources'];indices=[i for i,s in enumerate(sources) if s['run']==OLD];assert len(indices)==1
    index=indices[0]
    source=dict(run=candidate.RID,selector='earliest standalone macro-F1 maximum',epoch=selected['epoch'],
        **{k:selected[k] for k in ['checkpoint','checkpoint_sha256','prediction_file','prediction_sha256']})
    plan=dict(date='2026-10-10',experiment_id=RID,question='Does the predeclared stronger-decay ConvNeXt member improve S83 when replacing its original Tiny member?',
        original_sources=sources,replacement_source=source,replaced_index=index,
        rule='Exactly one equal-six replacement of S06 with standalone macro-F1-selected S95; all other members including CBAM unchanged',
        precision='Existing FP32 identity source probabilities; same float32 arithmetic mean as S83',weights=[1/6]*6,
        protocol=c['protocol'],split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
        reference_predictions=relative(REFERENCE/'validation_predictions.csv'),reference_sha256=sha256(REFERENCE/'validation_predictions.csv'),
        cache_provenance=sig['fp32_reference_caches'],predeclared_training_plan_sha256=sha256(candidate.BASE/'PREDECLARED_PLAN.json'),
        material_gate=candidate.config()['material_gate'],maximum_fusions=1,
        caveat='Repeated exploratory validation selection after earlier test outcomes were known; not independent test evidence. S95 uses FP32 validation, unlike original S06 AMP; historical comparison is not a pure causal weight-decay estimate.',
        gpu_used=False,test_loaded=False,training_epochs=0)
    write_json(OUT/'PREDECLARED_PLAN.json',plan)
    parts=fp32_parts(sig,val)
    cbam=aligned(ROOT/sources[-1]['prediction_file'],val).astype(np.float32)
    parts=parts+[cbam];base=np.mean(parts,axis=0)
    saved=aligned(REFERENCE/'validation_predictions.csv',val)
    np.testing.assert_allclose(base,saved,atol=1e-7,rtol=0)
    bm=report(y,base);assert bm['accuracy']==.9401197604790419
    assert abs(bm['macro_f1']-.8952008370253541)<1e-12
    p95=aligned(ROOT/source['prediction_file'],val).astype(np.float32)
    newparts=list(parts);newparts[index]=p95;p=np.mean(newparts,axis=0)
    m=package(OUT,val,p,'S96 fixed stronger-decay Tiny replacement')
    old=base.argmax(1);new=p.argmax(1);single=p95.argmax(1)
    gain=(old!=y)&(new==y);loss=(old==y)&(new!=y)
    gained=int(gain.sum());lost=int(loss.sum());net=gained-lost
    balanced=m['accuracy']>bm['accuracy'] and m['macro_f1']>=bm['macro_f1'] and m['per_class']['mel']['recall']>=bm['per_class']['mel']['recall']
    material=balanced and net>=8 and m['accuracy']-bm['accuracy']>=.005-1e-12
    result=dict(status='completed',**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']},
        reference_accuracy=bm['accuracy'],reference_macro_f1=bm['macro_f1'],gained=gained,lost=lost,net_correct=net,
        correct=int((new==y).sum()),incorrect=int((new!=y).sum()),melanoma_recall=m['per_class']['mel']['recall'],
        reference_melanoma_recall=bm['per_class']['mel']['recall'],akiec_recall=m['per_class']['akiec']['recall'],
        standalone_fixes_reference_errors=int(((old!=y)&(single==y)).sum()),
        standalone_errors_on_reference_correct=int(((old==y)&(single!=y)).sum()),balanced_gain=bool(balanced),material_gate_passed=bool(material),
        decision='material_gain_candidate' if material else ('directional_only' if balanced else 'reject_no_balanced_gain'),
        gpu_used=False,test_loaded=False,runtime_seconds=time.perf_counter()-started)
    changes=[dict(image_id=val.iloc[i].image_id,true_class=CLASSES[y[i]],reference_class=CLASSES[old[i]],candidate_class=CLASSES[new[i]],
        change='gained' if gain[i] else 'lost') for i in np.flatnonzero(gain|loss)]
    write_csv(OUT/'changed_predictions.csv',changes,['image_id','true_class','reference_class','candidate_class','change'])
    class_rows=[]
    for i,cl in enumerate(CLASSES):
        mask=y==i;class_rows.append(dict(class_name=cl,reference_correct=int(((old==y)&mask).sum()),
            ensemble_correct=int(((new==y)&mask).sum()),gained=int((gain&mask).sum()),lost=int((loss&mask).sum()),
            standalone_recovers_reference_errors=int(((old!=y)&(single==y)&mask).sum()),**m['per_class'][cl]))
    write_csv(OUT/'class_complementarity.csv',class_rows)
    comparison_figures([dict(display_name='S83 existing equal-six',accuracy=bm['accuracy'],macro_f1=bm['macro_f1']),
        dict(display_name='S96 S95 replaces S06 (equal-six)',accuracy=m['accuracy'],macro_f1=m['macro_f1'])],OUT/'comparison_figures',
        'Stronger-decay member replacement | exploratory validation')
    independent=report(y,aligned(OUT/'validation_predictions.csv',val))
    assert independent['confusion_matrix']==m['confusion_matrix'] and abs(independent['macro_f1']-m['macro_f1'])<1e-12
    cm=pd.read_csv(OUT/'figures/confusion_matrix.csv',index_col=0).to_numpy()
    np.testing.assert_array_equal(cm,m['confusion_matrix'])
    assert result['correct']==1413+net
    assert sha256(REFERENCE/'validation_predictions.csv')==plan['reference_sha256']
    assert all(sha256(ROOT/s['checkpoint'])==s['checkpoint_sha256'] for s in sources+[source])
    assert sha256(ROOT/source['prediction_file'])==source['prediction_sha256']
    assert all(sha256(ROOT/s['path'])==s['sha256'] for s in sig['fp32_reference_caches'])
    write_json(OUT/'verification.json',dict(status='passed',samples=1503,reference_fp32_mean_reproduced=True,
        saved_metrics_recomputed=True,probability_formula_verified=True,confusion_class_scores_present=True,
        source_checkpoints_predictions_caches_unchanged=True,gpu_used=False,test_loaded=False))
    write_json(OUT/'summary.json',result)
    members=[source if i==index else s for i,s in enumerate(sources)]
    upsert(dict(experiment_id=RID,era='structured',record_kind='fixed_probability_fusion',phase='post_test_exploratory_development',
        protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
        method=plan['rule'],model='S83 equal-six with stronger-decay Tiny replacement',attention='Existing CBAM source preserved',
        image_size=224,seed=42,epochs=0,ensemble_members=json.dumps([s['run'] for s in members]),ensemble_weights=json.dumps([1/6]*6),
        status='completed',decision=result['decision'],config_path=relative(OUT/'PREDECLARED_PLAN.json'),metrics_path=relative(OUT/'validation_metrics.json'),
        source_sha256=sha256(OUT/'validation_metrics.json'),plots_dir=relative(OUT/'figures'),confusion_matrix_path=relative(OUT/'figures/confusion_matrix.csv'),
        runtime_seconds=result['runtime_seconds'],val_loss=m['loss'],notes=plan['caveat'],**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    final={r['experiment_id']:r for r in read_registry()}
    assert all(r==final[r['experiment_id']] for r in committed if r['experiment_id']!=candidate.RID)
    s=json.loads((candidate.OUT/'summary.json').read_text());ref=json.loads((ROOT/'results/structured_experiments'/OLD/'validation_metrics.json').read_text())
    text=f'''# S95/S96 regularization closeout

10 October 2026. S95 completed20epochs. CPU-only closeout verified all best-accuracy/best-F1/latest checkpoints, finite model/optimizer states, strict model loading, saved prediction/history metrics, class scores, matrices and curves. No new GPU inference or test scoring. Source code/config/pretraining fingerprints remain unchanged.

| Standalone | Best accuracy | Macro-F1 at accuracy winner | Best macro-F1 | Epoch |
|---|---:|---:|---:|---:|
| S06 historical control | {100*ref['accuracy']:.4f}% | {ref['macro_f1']:.6f} | 0.868746 | accuracy17 / F118 |
| S95 decay0.05 | {100*s['best_accuracy']:.4f}% | {s['best_macro_f1']:.6f} | {s['best_macro_f1']:.6f} | both{s['best_epoch']} |

Final20: {100*s['final_metrics']['accuracy']:.4f}% / {s['final_metrics']['macro_f1']:.6f}. Stronger decay did not improve standalone performance. Historical AMP versus new FP32 validation and the modern numerical safeguards limit a pure causal claim; report this as a negative development experiment.

| Fixed ensemble | Accuracy | Macro-F1 | Gained / lost vs S83 | Melanoma recall |
|---|---:|---:|---:|---:|
| S83 existing equal-six | {100*bm['accuracy']:.4f}% | {bm['macro_f1']:.6f} | — | {100*bm['per_class']['mel']['recall']:.2f}% |
| S96 S95 replaces original Tiny | {100*m['accuracy']:.4f}% | {m['macro_f1']:.6f} | {gained} / {lost} | {100*m['per_class']['mel']['recall']:.2f}% |

Net {net:+d} correct; material gate passed: {material}. Decision: {result['decision']}. S95 alone recovers {result['standalone_fixes_reference_errors']} of S83's90errors, but makes {result['standalone_errors_on_reference_correct']} errors on previously correct images; these are diagnostics, not an oracle ensemble.

All other members/weights and CBAM stayed fixed. Exactly one predeclared replacement used S95's existing standalone F1 winner; no weights, epochs or alternative combinations searched. Existing S83 and historical rows/checkpoints are preserved. All unrelated committed registry rows were verified unchanged.

Artifacts: `results/short_screening/convnext_regularization_v1/{candidate.RID}/` (training/full20verification), matching local `checkpoints/short_screening/convnext_regularization_v1/` folder; `results/short_screening/s96_convnext_replacement_cpu/` (plan/source hashes, metrics, prediction probabilities, changed IDs, class scores, confusion/comparison PNG/PDF figures, verification).

{plan['caveat']} A later evaluation on the previously evaluated test cohort is a post-development audit, not a new first independent test. Reaching95% validation is not guaranteed and cannot determine whether unseen-test accuracy exceeds95%.
'''
    atomic_text(ROOT/'research/short_screening/S95_S96_REGULARIZATION_CLOSEOUT.md',text)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
