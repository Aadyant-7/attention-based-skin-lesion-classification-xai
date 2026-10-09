"""Exactly two output-only stability interventions on S83; no parameter search."""
import json
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from research.common import ROOT, CLASSES, relative, sha256, write_json, write_csv, atomic_text
from research.final_cbam_development.runtime import require_launch_freeze
from research.final_cbam_development.protocol import verified_development
from research.final_cbam_development.artifacts import fp32_parts, COLS
from research.short_screening.lesion_bag_screen import report
from research.short_screening.s82_cbam_addition_cpu import package
from research.plots import comparison_figures
from research.registry import upsert

OUT=ROOT/'results/short_screening/s84_s85_cbam_stability_cpu'


def checked(frame,val):
    assert frame.image_id.tolist()==val.image_id.tolist() and frame.true_class.tolist()==val.diagnosis.tolist()
    p=frame[COLS].to_numpy(dtype=np.float64)
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
    return p


def temper(p,temperature=2.0):
    if temperature==1:return p.copy()
    return softmax(np.log(np.maximum(p,1e-12))/temperature,axis=1)


def ece(y,p):
    confidence=p.max(1);correct=p.argmax(1)==y;value=0.
    for i in range(10):
        lo=i/10;hi=(i+1)/10;mask=(confidence>=lo)&((confidence<hi) if i<9 else (confidence<=hi))
        if mask.any():value+=float(mask.mean())*abs(float(correct[mask].mean())-float(confidence[mask].mean()))
    return value


def main():
    if (OUT/'summary.json').exists():print('Completed two-method study preserved');return
    start=time.perf_counter();c,sig=require_launch_freeze();_,val=verified_development(c)
    f1source=ROOT/c['results']/'macro_f1_selected/validation_predictions.csv';accsource=ROOT/c['results']/'validation_predictions.csv'
    f1=checked(pd.read_csv(f1source),val);acc=checked(pd.read_csv(accsource),val)
    sources=sig['phase1']['ensemble']['sources']+[dict(run=c['experiment_id'],selector='standalone_macro_f1',epoch=35,
        checkpoint=c['checkpoints']+'/best_macro_f1.pt',checkpoint_sha256=sha256(ROOT/c['checkpoints']/'best_macro_f1.pt'),
        prediction_file=relative(f1source),prediction_sha256=sha256(f1source))]
    extra=dict(run=c['experiment_id'],selector='standalone_accuracy',epoch=33,checkpoint=c['checkpoints']+'/best.pt',
        checkpoint_sha256=sha256(ROOT/c['checkpoints']/'best.pt'),prediction_file=relative(accsource),prediction_sha256=sha256(accsource))
    plan=dict(date='2026-10-09',reference='S83 fixed equal-six, macro-F1-selected CBAM35',sources=sources,additional_source=extra,
        protocol=c['protocol'],split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
        methods=[dict(id=84,name='cbam_only_fixed_temperature',rule='CBAM35 softmax(log(max(p,1e-12))/2); others unchanged; equal-six',
            question='Does a fixed reduction of the newly added, unusually overconfident CBAM scores reduce domination by confident errors?',
            caveat='Fixed confidence tempering, not a fitted or guaranteed calibration; earlier all-five fitted calibration was rejected'),
            dict(id=85,name='cbam_two_winner_probability_bag',rule='Average CBAM33 and35 probabilities first; this remains one1/6 CBAM slot; other five1/6 each',
            question='Does prediction averaging of the two already-saved standalone selection winners reduce CBAM late-epoch variance?',
            caveat='Two selected checkpoint prediction bagging; not parameter averaging or a cyclic-learning-rate snapshot-ensemble reproduction')],
        comparisons='Both against S83 and historical S53; report all, no follow-up temperatures/weights/epochs',
        gate=dict(minimum_net_correct=8,minimum_accuracy_gain=.005,macro_f1_nondecreasing=True,melanoma_recall_nondecreasing=True),
        scope='Repeated exploratory validation after known test outcomes; no independent test or causal generalization claim',
        test_loaded=False,gpu_used=False,training_epochs=0)
    if (OUT/'PREDECLARED_PLAN.json').exists():assert json.loads((OUT/'PREDECLARED_PLAN.json').read_text())==plan
    else:write_json(OUT/'PREDECLARED_PLAN.json',plan)
    write_json(OUT/'input_provenance.json',dict(fp32_reference_caches=sig['fp32_reference_caches'],
        cache_signature_sha256=sig['s76_signature_sha256'],
        note='The original five sources use verified fresh FP32 S76 caches, not their earlier AMP prediction CSVs.',
        cbam_probability_sources=[sources[-1],extra]))
    parts=[p.astype(np.float64) for p in fp32_parts(sig,val)];stack=np.stack(parts+[f1]);y=val.label.to_numpy();base=stack.mean(0)
    saved=pd.read_csv(ROOT/'results/short_screening/s83_cbam_f1_addition_cpu/validation_probabilities.csv')[COLS].to_numpy()
    assert np.allclose(base,saved,atol=1e-7) and np.array_equal(base.argmax(1),saved.argmax(1))
    reference=report(y,base);s53=report(y,np.mean(parts,axis=0));assert reference['accuracy']==.9401197604790419
    votes=stack.argmax(2)==y;wrong=base.argmax(1)!=y
    diagnostic=dict(reference_errors=int(wrong.sum()),all_six_wrong=int((wrong&~votes.any(0)).sum()),
        correct_member_count_on_errors=np.bincount(votes.sum(0)[wrong],minlength=7).tolist(),
        members=[dict(run=m['run'],fixes_reference_errors=int((correct&wrong).sum()),
            mean_confidence_when_wrong=float(p.max(1)[~correct].mean())) for m,p,correct in zip(sources,stack,votes)],
        caveat='Label-based error diagnostic, not a deployable oracle score or routing rule')
    write_json(OUT/'error_diagnostic.json',diagnostic)
    softened=temper(f1);assert np.array_equal(softened.argmax(1),f1.argmax(1))
    alternatives=[np.mean(parts+[softened],axis=0),np.mean(parts+[(acc+f1)/2],axis=0)]
    rows=[dict(display_name='S83 equal-six reference',accuracy=reference['accuracy'],macro_f1=reference['macro_f1'])];results={}
    old=base.argmax(1);bc=old==y
    for specification,p in zip(plan['methods'],alternatives):
        path=OUT/specification['name'];m=package(path,val,p,'S'+str(specification['id'])+' '+specification['name'])
        correct=p.argmax(1)==y;gain=int((~bc&correct).sum());lost=int((bc&~correct).sum());net=gain-lost
        passed=net>=8 and m['accuracy']-reference['accuracy']>=.005-1e-12 and m['macro_f1']>=reference['macro_f1'] and m['per_class']['mel']['recall']>=reference['per_class']['mel']['recall']
        gate_s53=(m['accuracy']-s53['accuracy']>=.005-1e-12 and int(correct.sum())-int((np.mean(parts,axis=0).argmax(1)==y).sum())>=8
            and m['macro_f1']>=s53['macro_f1'] and m['per_class']['mel']['recall']>=s53['per_class']['mel']['recall'])
        m['ten_bin_ece']=ece(y,p);write_json(path/'validation_metrics.json',m)
        changes=[]
        for i,row in val.iterrows():
            if correct[i]!=bc[i]:changes.append(dict(image_id=row.image_id,true_class=row.diagnosis,reference_class=CLASSES[old[i]],candidate_class=CLASSES[p[i].argmax()],change='gained' if correct[i] else 'lost'))
        write_csv(path/'changed_predictions.csv',changes,['image_id','true_class','reference_class','candidate_class','change'])
        result=dict(accuracy=m['accuracy'],macro_f1=m['macro_f1'],macro_precision=m['macro_precision'],macro_recall=m['macro_recall'],
            gained=gain,lost=lost,net_correct=net,correct=int(correct.sum()),incorrect=int((~correct).sum()),accuracy_gain_pp=100*(m['accuracy']-reference['accuracy']),
            melanoma_recall=m['per_class']['mel']['recall'],nll=m['loss'],ece=m['ten_bin_ece'],
            material_gate_vs_s83=bool(passed),material_gate_vs_s53=bool(gate_s53),
            decision='material_gain_candidate' if passed else ('directional_candidate_only' if m['accuracy']>reference['accuracy'] else 'reject_no_accuracy_gain'))
        results[specification['name']]=result;write_json(path/'summary.json',result)
        write_json(path/'verification.json',dict(status='passed',samples=1503,probabilities_finite_normalized=True,metrics_recomputed=True,test_loaded=False,gpu_used=False))
        rows.append(dict(display_name='S'+str(specification['id'])+' '+specification['name'],accuracy=m['accuracy'],macro_f1=m['macro_f1']))
        upsert(dict(experiment_id=f"s{specification['id']}_{specification['name']}_exploratory_seed42",era='structured',record_kind='fixed_probability_fusion',phase='post_final_development',
            protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],method=specification['rule'],
            attention='CBAM in S79 only',model='convnext_tiny+convnext_small+densenet201+efficientnet_v2_s+efficientnet_b0+convnext_tiny_cbam',
            ensemble_members=json.dumps([m['run'] for m in sources]),ensemble_weights=json.dumps([1/6]*6),
            preprocessing='FP32 identity probabilities; 224x224 source preprocessing',
            source_selector='CBAM35 fixed T=2' if specification['id']==84 else 'CBAM33/35 averaged within one 1/6 slot',
            val_loss=m['loss'],confusion_matrix_path=relative(path/'figures/confusion_matrix.csv'),
            image_size=224,seed=42,epochs=0,status='completed',decision=result['decision'],
            config_path=relative(OUT/'PREDECLARED_PLAN.json'),metrics_path=relative(path/'validation_metrics.json'),plots_dir=relative(path/'figures'),
            notes=plan['scope']+'; '+specification['caveat'],**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    comparison_figures(rows,OUT/'comparison_figures','Two fixed CBAM stability changes | exploratory validation')
    assert all(sha256(ROOT/a['checkpoint'])==a['checkpoint_sha256'] and sha256(ROOT/a['prediction_file'])==a['prediction_sha256'] for a in sources+[extra])
    summary=dict(status='completed',reference_accuracy=reference['accuracy'],reference_macro_f1=reference['macro_f1'],reference_melanoma_recall=reference['per_class']['mel']['recall'],
        reference_nll=reference['loss'],reference_ece=ece(y,base),results=results,source_hashes_unchanged=True,runtime_seconds=time.perf_counter()-start,
        decision='Two-method budget complete; preserve all; no automatic weight/temperature/checkpoint sweep or test run',test_loaded=False,gpu_used=False)
    write_json(OUT/'summary.json',summary)
    text='# S84 S85 fixed CBAM stability closeout\n\nExactly two CPU-only interventions were recorded before fusion/scoring. Source checkpoints and input probabilities are unchanged. No training, new inference or test labels/images.\n\n'
    text+='| Method | Accuracy | Macro-F1 | Melanoma recall | Net correct vs S83 |\n|---|---:|---:|---:|---:|\n'
    text+=f"| S83 equal-six reference | {100*reference['accuracy']:.4f}% | {reference['macro_f1']:.6f} | {100*reference['per_class']['mel']['recall']:.2f}% | 0 |\n"
    for name,r in results.items():text+=f"| {name} | {100*r['accuracy']:.4f}% | {r['macro_f1']:.6f} | {100*r['melanoma_recall']:.2f}% | {r['net_correct']:+d} |\n"
    text+='\nS84 uses one fixed temperature2 on the CBAM branch only; this does not change its standalone argmax and is not a fitted calibration. S85 averages the saved standalone accuracy33 and macro-F1 35 probabilities within one CBAM slot; it does not increase the CBAM branch weight or average model parameters. Other five sources stay1/6.\n\n'
    text+='The diagnostic found68/90 S83 errors had at least one correct component,22had no correct component argmax, and CBAM wrong-prediction confidence averaged0.9241. These label-based facts motivate checks, not deployable oracle selection or proof that all these cases can be recovered.\n\n'
    text+='Material gates remain>=0.005 accuracy and8netcorrect with macro-F1/melanoma recall nondecrease; summaries separately show gates against S83 and S53. Directional gains below the gate remain exploratory candidates. Reused validation/known earlier test outcomes limit independent performance claims. Both full prediction/probability/class/confusion/comparison PNG/PDF packages and changed identities are saved under `results/short_screening/s84_s85_cbam_stability_cpu/`. No further parameter search or GPU run is authorized by this script.\n'
    atomic_text(ROOT/'research/short_screening/S84_S85_CBAM_STABILITY_CLOSEOUT.md',text);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
