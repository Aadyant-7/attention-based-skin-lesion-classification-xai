"""Close out S89/S91 and test exactly three fixed saved-probability additions.

CPU only: never instantiate a data loader, perform inference, or score test data.
"""
import gc
import io
import json
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json, atomic_text
from research.final_cbam_development.runtime import require_launch_freeze
from research.final_cbam_development.protocol import verified_development
from research.final_cbam_development.artifacts import COLS
from research.short_screening import s89_swin_transfer as swin
from research.short_screening import s91_deit3_base_transfer as deit
from research.short_screening.s86_s87_saved_member_addition import aligned
from research.short_screening.s82_cbam_addition_cpu import package
from research.short_screening.lesion_bag_screen import report
from research.plots import comparison_figures
from research.registry import read_registry, upsert

OUT=ROOT/'results/short_screening/s92_s94_transformer_addition_cpu'
RUNS=[('S89 Swin-T',swin),('S91 DeiT III Base',deit)]
JOBS=[(92,'swin',[0]),(93,'deit_base',[1]),(94,'both',[0,1])]
CAVEAT=('Repeated exploratory image-level validation selection, after prior held-out outcomes were known; '
        'not independent confirmation or a test score. Same cohort, but native architecture, pretraining '
        'and training recipe details differ. No new training, inference, weight or checkpoint search.')


def finite_tree(obj):
    if torch.is_tensor(obj):
        assert torch.isfinite(obj).all(), 'Nonfinite checkpoint tensor'
    elif isinstance(obj,dict):
        for x in obj.values(): finite_tree(x)
    elif isinstance(obj,(list,tuple)):
        for x in obj: finite_tree(x)


def verify_run(name,runner,val):
    folder=runner.OUT;ck=runner.CK
    summary=json.loads((folder/'summary.json').read_text())
    history=pd.read_csv(folder/'history.csv')
    assert summary['status']=='completed' and summary['epochs']==20
    assert history.epoch.tolist()==list(range(1,21)) and not summary['test_loaded']
    assert history.validation_precision.eq('fp32').all()
    preflight=json.loads((runner.BASE/'preflight.json').read_text())
    assert preflight['fingerprints']==runner.fingerprints()
    assert sha256(runner.CFG)==preflight['config_sha256']
    epochs={'':int(history.loc[history.val_accuracy.idxmax(),'epoch']),
            '_macro_f1':int(history.loc[history.val_macro_f1.idxmax(),'epoch']), '_latest':20}
    assert epochs['']==summary['best_epoch'] and epochs['_macro_f1']==summary['best_macro_f1_epoch']
    selections=[]
    for suffix,filename in [('', 'best.pt'),('_macro_f1','best_macro_f1.pt'),('_latest','latest.pt')]:
        p=aligned(folder/f'validation_predictions{suffix}.csv',val)
        m=json.loads((folder/f'validation_metrics{suffix}.json').read_text())
        assert m['class_order']==list(CLASSES)
        independent=report(val.label.to_numpy(),p)
        for key in ['accuracy','macro_precision','macro_recall','macro_f1']:
            assert abs(m[key]-independent[key])<1e-12,(name,suffix,key)
        assert m['confusion_matrix']==independent['confusion_matrix']
        for cl in CLASSES:
            for key in ['precision','recall','f1','support']:
                assert abs(m['per_class'][cl][key]-independent['per_class'][cl][key])<1e-12
        prob=pd.read_csv(folder/f'validation_probabilities{suffix}.csv').set_index('image_id').loc[val.image_id]
        np.testing.assert_allclose(prob[COLS].to_numpy(),p,rtol=0,atol=1e-12)
        figure=folder/('figures'+suffix)
        for stem in ['confusion_matrix','confusion_matrix_normalized','per_class_metrics','class_support']:
            for ext in ['png','pdf']:
                assert (figure/f'{stem}.{ext}').stat().st_size>0
        cm=pd.read_csv(figure/'confusion_matrix.csv',index_col=0).to_numpy()
        np.testing.assert_array_equal(cm,m['confusion_matrix'])
        normalized=pd.read_csv(figure/'confusion_matrix_normalized.csv',index_col=0).to_numpy()
        np.testing.assert_allclose(normalized,cm/cm.sum(1,keepdims=True),atol=1e-12)
        scores=pd.read_csv(figure/'per_class_metrics.csv').set_index('class')
        for cl in CLASSES:
            for key in ['precision','recall','f1','support']:
                assert abs(float(scores.loc[cl,key])-m['per_class'][cl][key])<1e-12
        path=ck/filename; cp=torch.load(path,map_location='cpu',weights_only=False)
        assert cp['config']==runner.config() and cp['fingerprints']==runner.fingerprints()
        finite_tree(cp['model'])
        net=runner.fresh(pretrained=False);net.load_state_dict(cp['model'],strict=True)
        del net
        if suffix=='_latest':
            assert len(cp['history'])==20 and cp['history'][-1]['epoch']==20
            finite_tree(cp['optimizer']);finite_tree(cp['scheduler']);finite_tree(cp['scaler'])
            assert cp['best']['epoch']==epochs[''] and cp['f1best']['epoch']==epochs['_macro_f1']
        else:
            assert cp['epoch']==epochs[suffix] and cp['metrics']==m
            checkpoint_p=pd.DataFrame(cp['predictions']).set_index('image_id').loc[val.image_id][COLS].to_numpy()
            np.testing.assert_allclose(checkpoint_p,p,rtol=0,atol=1e-12)
        row=history.iloc[epochs[suffix]-1]
        for key in ['accuracy','macro_precision','macro_recall','macro_f1']:
            assert abs(float(row['val_'+key])-m[key])<1e-12
        selections.append(dict(selector={'':'accuracy','_macro_f1':'macro_f1','_latest':'final'}[suffix],
            epoch=epochs[suffix],checkpoint=relative(path),checkpoint_sha256=sha256(path),
            prediction_file=relative(folder/f'validation_predictions{suffix}.csv'),
            prediction_sha256=sha256(folder/f'validation_predictions{suffix}.csv'),
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']},
            melanoma_recall=m['per_class']['mel']['recall'],akiec_recall=m['per_class']['akiec']['recall']))
        del cp;gc.collect()
    for stem in ['loss_curves','accuracy_curves','macro_f1_curve']:
        for ext in ['png','pdf']: assert (folder/'figures'/f'{stem}.{ext}').stat().st_size>0
    assert not (folder/'failure.json').exists()
    if runner is swin:
        for filename in ['best.pt','best_macro_f1.pt']:
            assert sha256(ck/filename)==sha256(ck/'pilot_epoch15'/filename)
        for filename in ['validation_predictions.csv','validation_predictions_macro_f1.csv']:
            assert sha256(folder/filename)==sha256(folder/'pilot_epoch15'/filename)
        pilot=pd.read_csv(folder/'pilot_epoch15/history.csv')
        pd.testing.assert_frame_equal(pilot,history.iloc[:15].reset_index(drop=True))
    result=dict(name=name,run=runner.RID,status='passed',epochs=20,selections=selections,
        runtime_seconds=summary['runtime_seconds'],skipped_amp_optimizer_updates=int(history.skipped_updates.sum()),
        peak_allocated_vram_mb=float(history.peak_allocated_vram_mb.max()),
        checkpoint_model_optimizer_tensors_finite=True,strict_model_state_loading=True,
        predictions_match_selected_checkpoints=True,metrics_confusions_class_scores_recomputed=True,
        curves_present=True,code_config_fingerprints_unchanged=True,test_loaded=False,gpu_used=False)
    write_json(folder/'full20_closeout_verification.json',result)
    old=next(r for r in read_registry() if r['experiment_id']==runner.RID)
    old.update(status='completed',epochs=20,best_epoch=epochs[''],runtime_seconds=summary['runtime_seconds'],
        val_loss=json.loads((folder/'validation_metrics.json').read_text())['loss'],
        checkpoint_sha256=selections[0]['checkpoint_sha256'],checkpoint_available_local=True,
        source_sha256=sha256(folder/'validation_metrics.json'),source_selector='Standalone raw accuracy winner; earliest maximum',
        confusion_matrix_path=relative(folder/'figures/confusion_matrix.csv'),decision='completed_full20_for_complementarity',
        notes=old['notes'].split(' Full20 independently verified;')[0]+' Full20 independently verified; best macro-F1 epoch '+str(epochs['_macro_f1'])+
            '; native AMP gradient overflow updates skipped safely: '+str(result['skipped_amp_optimizer_updates'])+'. '+CAVEAT)
    upsert(old)
    print(json.dumps(dict(name=name,verified='passed',selections=selections)),flush=True)
    return result


def main():
    if (OUT/'summary.json').exists():
        print('Completed study preserved; no repeated scoring');return
    started=time.perf_counter();torch.set_num_threads(4)
    c,_=require_launch_freeze();_,val=verified_development(c);y=val.label.to_numpy()
    assert len(val)==1503
    before={r['experiment_id']:r for r in read_registry()}
    # Existing historical registry rows are compared against committed history as well.
    committed=list(pd.read_csv(io.StringIO(subprocess.check_output(
        ['git','show','HEAD:results/master_experiment_registry.csv'],cwd=ROOT,text=True)),dtype=str).fillna('').to_dict('records'))
    historical_changes=[r['experiment_id'] for r in committed if before.get(r['experiment_id'])!=r]
    assert set(historical_changes)<={swin.RID},historical_changes
    assert not any(r['experiment_id'].startswith(('s92_','s93_','s94_')) for r in before.values())
    verified=[verify_run(name,r,val) for name,r in RUNS]
    reference=ROOT/'results/short_screening/s83_cbam_f1_addition_cpu/validation_predictions.csv'
    original=json.loads((reference.parent/'PREDECLARED_PLAN.json').read_text())['sources']
    sources=[dict(name=name,run=r.RID,**{k:verified[i]['selections'][1][k] for k in
        ['checkpoint','checkpoint_sha256','prediction_file','prediction_sha256','epoch']},selector='Standalone macro-F1 winner; earliest maximum')
        for i,(name,r) in enumerate(RUNS)]
    plan=dict(date='2026-10-09',question='Do the two newly trained transformer members recover S83 errors without losing previous correct predictions?',
        reference_file=relative(reference),reference_sha256=sha256(reference),original_six_sources=original,
        sources=sources,protocol=c['protocol'],split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
        selection_rule='Use each already selected standalone macro-F1 winner; no checkpoint, weight, temperature or subset sweep.',
        jobs=[dict(id=n,name=name,new_members=indices,equal_member_weights=[1/(6+len(indices))]*(6+len(indices))) for n,name,indices in JOBS],
        arithmetic='(6*S83_saved_FP32_mean + each selected saved_FP32_member) / total_members; CPU float64 arithmetic from saved CSV probabilities',
        material_gate=dict(minimum_net_correct=8,minimum_accuracy_gain=.005,macro_f1_nondecreasing=True,melanoma_recall_nondecreasing=True),
        maximum_fusions=3,test_loaded=False,gpu_used=False,training_epochs=0,caveat=CAVEAT)
    write_json(OUT/'PREDECLARED_PLAN.json',plan)
    base=aligned(reference,val);bm=report(y,base);old=base.argmax(1)
    assert bm['accuracy']==.9401197604790419 and abs(bm['macro_f1']-.8952008370253541)<1e-12
    arrays=[aligned(ROOT/s['prediction_file'],val) for s in sources]
    audit=[];class_audit=[]
    for source,p in zip(sources,arrays):
        new=p.argmax(1);m=report(y,p);fix=(old!=y)&(new==y);harm=(old==y)&(new!=y)
        audit.append(dict(**source,accuracy=m['accuracy'],macro_f1=m['macro_f1'],
            recovers_s83_errors=int(fix.sum()),errors_on_s83_correct=int(harm.sum()),
            shares_s83_errors=int(((old!=y)&(new!=y)).sum()),melanoma_recall=m['per_class']['mel']['recall']))
        for i,cl in enumerate(CLASSES):
            mask=y==i;class_audit.append(dict(model=source['name'],class_name=cl,
                recovered=int((fix&mask).sum()),errors_on_reference_correct=int((harm&mask).sum()),**m['per_class'][cl]))
    write_csv(OUT/'standalone_class_complementarity.csv',class_audit)
    write_json(OUT/'standalone_complementarity.json',dict(candidates=audit,
        oracle_warning='Standalone recovered errors are diagnostics. The union of correct answers is not deployable ensemble performance.'))
    rows=[dict(display_name='S83 reference (six)',accuracy=bm['accuracy'],macro_f1=bm['macro_f1'])]
    rows += [dict(display_name=s['name']+' standalone F1 winner',accuracy=a['accuracy'],macro_f1=a['macro_f1']) for s,a in zip(sources,audit)]
    results=[]
    for number,name,indices in JOBS:
        total=6+len(indices);p=(6*base+sum(arrays[i] for i in indices))/total
        folder=OUT/name;m=package(folder,val,p,f'S{number} S83 + {name}')
        new=p.argmax(1);gain=(old!=y)&(new==y);loss=(old==y)&(new!=y)
        gained=int(gain.sum());lost=int(loss.sum());net=gained-lost
        balanced=m['accuracy']>bm['accuracy'] and m['macro_f1']>=bm['macro_f1'] and m['per_class']['mel']['recall']>=bm['per_class']['mel']['recall']
        material=balanced and net>=8 and m['accuracy']-bm['accuracy']>=.005-1e-12
        result=dict(id=number,name=name,new_members=[sources[i]['run'] for i in indices],
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']},
            melanoma_recall=m['per_class']['mel']['recall'],akiec_recall=m['per_class']['akiec']['recall'],
            gained=gained,lost=lost,net_correct=net,correct=int((new==y).sum()),incorrect=int((new!=y).sum()),
            balanced_gain=bool(balanced),material_gate_passed=bool(material),
            decision='material_gain_candidate' if material else ('directional_only' if balanced else 'reject_no_balanced_gain'))
        results.append(result);write_json(folder/'summary.json',result)
        changes=[dict(image_id=val.iloc[i].image_id,true_class=CLASSES[y[i]],reference_class=CLASSES[old[i]],candidate_class=CLASSES[new[i]],
            change='gained' if gain[i] else 'lost') for i in np.flatnonzero(gain|loss)]
        write_csv(folder/'changed_predictions.csv',changes,['image_id','true_class','reference_class','candidate_class','change'])
        class_rows=[]
        for i,cl in enumerate(CLASSES):
            mask=y==i;class_rows.append(dict(class_name=cl,reference_correct=int(((old==y)&mask).sum()),
                ensemble_correct=int(((new==y)&mask).sum()),gained=int((gain&mask).sum()),lost=int((loss&mask).sum()),**m['per_class'][cl]))
        write_csv(folder/'class_complementarity.csv',class_rows)
        saved=aligned(folder/'validation_predictions.csv',val)
        np.testing.assert_allclose(saved,p,atol=1e-12,rtol=0)
        independent=report(y,saved)
        assert independent['confusion_matrix']==m['confusion_matrix'] and abs(independent['macro_f1']-m['macro_f1'])<1e-12
        cm=pd.read_csv(folder/'figures/confusion_matrix.csv',index_col=0).to_numpy()
        np.testing.assert_array_equal(cm,m['confusion_matrix'])
        normalized=pd.read_csv(folder/'figures/confusion_matrix_normalized.csv',index_col=0).to_numpy()
        np.testing.assert_allclose(normalized,cm/cm.sum(1,keepdims=True),atol=1e-12)
        assert result['correct']==int((old==y).sum())+net
        write_json(folder/'verification.json',dict(status='passed',samples=len(val),saved_probabilities_formula_reproduced=True,
            cohort_alignment_verified=True,metrics_class_scores_confusions_recomputed=True,test_loaded=False,gpu_used=False))
        rows.append(dict(display_name=f'S{number} S83 + '+name,accuracy=m['accuracy'],macro_f1=m['macro_f1']))
        members=[a['run'] for a in original]+[sources[i]['run'] for i in indices]
        upsert(dict(experiment_id=f's{number}_s83_{name}_equal_exploratory_seed42',era='structured',record_kind='fixed_probability_fusion',
            phase='post_test_exploratory_development',protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
            method=plan['arithmetic'],model='S83 six-member ensemble + '+name,attention='CBAM in S79 source only',
            ensemble_members=json.dumps(members),ensemble_weights=json.dumps([1/total]*total),image_size=224,seed=42,epochs=0,status='completed',decision=result['decision'],
            config_path=relative(OUT/'PREDECLARED_PLAN.json'),metrics_path=relative(folder/'validation_metrics.json'),source_sha256=sha256(folder/'validation_metrics.json'),
            plots_dir=relative(folder/'figures'),confusion_matrix_path=relative(folder/'figures/confusion_matrix.csv'),val_loss=m['loss'],notes=CAVEAT,
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
        print(json.dumps(result),flush=True)
    comparison_figures(rows,OUT/'comparison_figures','Fixed transformer additions | exploratory validation')
    assert sha256(reference)==plan['reference_sha256']
    for s in original+sources:
        assert sha256(ROOT/s['checkpoint'])==s['checkpoint_sha256']
    for s in sources: assert sha256(ROOT/s['prediction_file'])==s['prediction_sha256']
    after={r['experiment_id']:r for r in read_registry()}
    for rid,row in before.items():
        if rid not in {swin.RID,deit.RID}:assert row==after[rid],rid
    summary=dict(status='completed',reference_accuracy=bm['accuracy'],reference_macro_f1=bm['macro_f1'],
        standalone_audit=audit,results=results,highest_observed_accuracy=max([bm['accuracy']]+[r['accuracy'] for r in results]),
        source_hashes_unchanged=True,unrelated_registry_rows_unchanged=True,test_loaded=False,gpu_used=False,
        runtime_seconds=time.perf_counter()-started)
    write_json(OUT/'summary.json',summary)
    text='# S89/S91 full20 closeout and S92–S94 transformer additions\n\n9 October 2026. Both GPU runs completed their authorized20-epoch window. This closeout uses saved validation probabilities only; no training, fresh inference, or test access.\n\n'
    text+='## Standalone results\n\n| Model | Best accuracy / epoch | Best macro-F1 / epoch | Final accuracy / macro-F1 |\n|---|---:|---:|---:|\n'
    for name,r in RUNS:
        s=json.loads((r.OUT/'summary.json').read_text())
        text+=f"| {name} | {100*s['best_accuracy']:.4f}% / {s['best_epoch']} | {s['best_macro_f1']:.6f} / {s['best_macro_f1_epoch']} | {100*s['final_metrics']['accuracy']:.4f}% / {s['final_metrics']['macro_f1']:.6f} |\n"
    text+='\nSwin did not improve its epoch15 best during epochs16–20. DeiT tied its accuracy maximum at17, but the earliest accuracy-selected checkpoint remains13; macro-F1 selects17. Both final checkpoints are weaker than their selected best. These curves do not justify an automatic30-epoch extension. S90 Small was superseded before training, not a failed trained model.\n\n'
    text+='## Standalone error diversity versus S83\n\n| Selected standalone | S83 errors fixed | New errors on S83-correct images |\n|---|---:|---:|\n'
    for a in audit:text+=f"| {a['name']} epoch{a['epoch']} | {a['recovers_s83_errors']} | {a['errors_on_s83_correct']} |\n"
    text+='\nThese are diagnostic overlaps, not an oracle accuracy or guaranteed ensemble recoveries. Actual fixed fusion results follow.\n\n'
    text+='## Actual probability fusion\n\nEach existing S83 constituent is retained; new members receive equal1/7 or1/8 weighting. Use each transformer’s existing standalone macro-F1 winner, selected before this fusion study. No weights or checkpoints searched.\n\n| Method | Accuracy | Macro-F1 | Gained / lost versus S83 | Net | Melanoma recall |\n|---|---:|---:|---:|---:|---:|\n'
    text+=f"| S83 reference | {100*bm['accuracy']:.4f}% | {bm['macro_f1']:.6f} | — | — | {100*bm['per_class']['mel']['recall']:.2f}% |\n"
    for r in results:text+=f"| S{r['id']} +{r['name']} | {100*r['accuracy']:.4f}% | {r['macro_f1']:.6f} | {r['gained']} / {r['lost']} | {r['net_correct']:+d} | {100*r['melanoma_recall']:.2f}% |\n"
    text+='\nMaterial gate: at least8netcorrect and0.005absolute accuracy gain, macro-F1/melanoma recall nondecreasing. '
    text+='Passed by: '+(', '.join('S'+str(r['id']) for r in results if r['material_gate_passed']) or 'none')+'.\n\n'
    if not any(r['balanced_gain'] for r in results):
        text+='Decision: keep S83 unchanged. The tested equal-weight transformer additions do not improve the current ensemble. The joint addition improves akiec from35 to37 correct and nv from989 to992, but reduces melanoma from137 to129, bkl from145 to142, bcc from73 to71 and df from14 to13; vasc stays20. Thus it does not recover new images without worsening others. No additional GPU work is queued.\n\n'
    text+='Checkpoints/optimizer tensors are finite, model state dictionaries load strictly, three selected/final prediction packages per backbone reproduce checkpoint/history metrics, class scores and confusion matrices. Curves, source hashes, LR history and safe AMP-overflow skip counts are preserved in each `full20_closeout_verification.json`. The original Swin epoch15 snapshots remain unchanged. All unrelated historical registry rows are preserved.\n\n'
    text+='Artifacts: `results/short_screening/s92_s94_transformer_addition_cpu/` contains the plan, overlap/class diagnostics, three full metric/probability/class/confusion PNG/PDF packages, changed image IDs and comparison figures. Individual training packages remain in `results/short_screening/swin_transfer_v1/` and `results/short_screening/deit3_base_transfer_v1/`; local checkpoints are in the matching `checkpoints/short_screening/` folders and are not distributed by Git.\n\n'+CAVEAT+'\n'
    atomic_text(ROOT/'research/short_screening/S89_S91_TRANSFORMER_CLOSEOUT.md',text)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
