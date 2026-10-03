"""Close S13 from saved probabilities on CPU; no image/GPU/test inference."""
import copy,json
import numpy as np
import pandas as pd
from research.common import ROOT,CLASSES,sha256,write_json,write_csv
from research.tta import check
from research.fuse import aligned_probabilities
from research.train import metric_report
from research.plots import comparison_figures
from research.registry import read_registry,upsert
from research.phase3.close_s02 import load,verify_predictions,verify_figures,verify_launch_code
ID='s13_s12_four_flip_tta_exploratory_seed42'
CONTROL='s13_s12_identity_fp32_control_exploratory_seed42'
S12='s12_s03_s06_s10_equal_probability_exploratory_seed42'
OUT=ROOT/'results/model_comparison/structured/s13_tta'

def paired(predictions,reference,truth):
    ok=predictions==truth;other=reference==truth
    return dict(correct=int(ok.sum()),fixed=int((ok&~other).sum()),broken=int((~ok&other).sum()),net_correct=int(ok.sum()-other.sum()),both_wrong=int((~ok&~other).sum()))

def main():
    path=ROOT/'results/structured_experiments'/ID
    config,record=load(path/'config.json'),load(path/'record.json')
    assert config==load(ROOT/'research/configs/phase3'/f'{ID}.json') and record['status']=='completed'
    frame,val,parents=check(config)
    signature=load(path/'inference_signature.json');assert signature['config']==config
    # Authenticate frozen S13 sources via its launch commit after future adapter additions.
    verify_launch_code({'code_hashes':{'research/tta.py':signature['runner_sha256'],**signature['supporting_source_hashes']}}, {'git_commit':'b91bfaf'})
    assert load(path/'environment.json')['precision']=='fp32'
    arrays=[];identities=[];hashes={}
    for rid in config['parent_run_ids']:
        views=[]
        for view in config['views']:
            f=path/f'{rid}_{view}.csv'
            views.append(aligned_probabilities(pd.read_csv(f),val));hashes[f.name]=sha256(f)
        identities.append(views[0]);arrays.append(np.mean(views,axis=0))
    counts=frame.query("split=='train'").diagnosis.value_counts()
    weights=np.sqrt(7009/np.array([counts[c] for c in CLASSES]));weights/=weights.mean()
    results={};predictions={};truth=val.diagnosis.to_numpy();labels=val.label.to_numpy()
    for name,suffix,values in [('identity_fp32','_identity_fp32',identities),('tta','',arrays)]:
        fused=sum(w*p for w,p in zip(config['weights'],values));metrics=load(path/f'validation_metrics{suffix}.json')
        f=path/f'validation_predictions{suffix}.csv';stored=pd.read_csv(f)
        probabilities=aligned_probabilities(stored,val)
        assert np.allclose(fused,probabilities,rtol=0,atol=1e-12)
        loss=float(np.average(-np.log(np.clip(fused[np.arange(len(val)),labels],1e-12,1)),weights=weights[labels]))
        assert np.isclose(metrics['loss'],loss,rtol=0,atol=1e-12)
        recomputed=metric_report(labels,fused,loss)
        for k in ('accuracy','macro_precision','macro_recall','macro_f1'):assert metrics[k]==recomputed[k]
        assert metrics['per_class']==recomputed['per_class']
        cm=verify_predictions(f,metrics,val)
        folder=path/('figures' if name=='tta' else 'figures_identity_fp32')
        verify_figures(folder,cm,metrics)
        predictions[name]=stored.set_index('image_id').loc[val.index].predicted_class.to_numpy()
        results[name]=metrics;hashes[f.name]=sha256(f);hashes[f'validation_metrics{suffix}.json']=sha256(path/f'validation_metrics{suffix}.json')
    rows=[];classes=[];refs={}
    for label,rid in [('S06 ConvNeXt','s06_convnext_tiny_none_exploratory_seed42'),('S09 previous triple','s09_s02_s03_s06_equal_probability_exploratory_seed42'),('S12 fixed triple',S12)]:
        p=ROOT/'results/structured_experiments'/rid;r=load(p/'record.json');m=load(p/'validation_metrics.json')
        assert r['protocol']==config['protocol'] and r['split_sha256']==config['split_sha256']
        verify_predictions(p/'validation_predictions.csv',m,val)
        refs[rid]=pd.read_csv(p/'validation_predictions.csv').set_index('image_id').loc[val.index].predicted_class.to_numpy()
        rows.append(dict(display_name=label,experiment_id=rid,**{k:m[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')},inference_model_passes=1 if rid.startswith('s06') else 3))
        classes.extend(dict(experiment_id=rid,class_name=c,**m['per_class'][c]) for c in CLASSES)
    for name,rid,label,cost in [('identity_fp32',CONTROL,'S12 FP32 identity control',3),('tta',ID,'S13 four-view TTA',12)]:
        m=results[name];rows.append(dict(display_name=label,experiment_id=rid,**{k:m[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')},inference_model_passes=cost))
        classes.extend(dict(experiment_id=rid,class_name=c,**m['per_class'][c]) for c in CLASSES)
    gains={name:{rid:paired(p,ref,truth) for rid,ref in refs.items()} for name,p in predictions.items()}
    gains['tta']['identity_fp32']=paired(predictions['tta'],predictions['identity_fp32'],truth)
    comparison_figures(rows,OUT,'Same exploratory validation | FP32 identity control vs fixed four-view TTA\nNo view/weight search; 3 vs 12 model passes')
    write_csv(OUT/'per_class_comparison.csv',classes)
    write_csv(OUT/'paired_predictions.csv',[dict(image_id=i,true_class=t,identity_fp32=predictions['identity_fp32'][j],tta=predictions['tta'][j],s12=refs[S12][j]) for j,(i,t) in enumerate(zip(val.index,truth))])
    before=read_registry();untouched=[r for r in before if r['experiment_id'] not in (ID,CONTROL)]
    for field in ('accuracy','macro_precision','macro_recall','macro_f1'):assert record[field]==results['tta'][field]
    assert record['source_sha256']==sha256(path/'validation_metrics.json')
    record.update(decision='reject_fixed_flip_tta_retain_s12_reference',notes=record['notes'].split(' Closed out:')[0]+' Closed out:12 view tables and2 aggregate prediction/metric/figure packages verified. TTA regresses; FP32 identity gains one correct image only. Preserve S12 reference; no TTA search.')
    upsert(record);write_json(path/'record.json',record)
    control=copy.deepcopy(record)
    control.update(experiment_id=CONTROL,record_kind='precision_control_inference',phase='inference_control',method='FP32 identity-only evaluation of unchanged S12; precision control, no TTA',decision='retain_precision_control_no_meaningful_accuracy_gain',runtime_seconds='',**{k:results['identity_fp32'][k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')},val_loss=results['identity_fp32']['loss'],metrics_path=(path/'validation_metrics_identity_fp32.json').relative_to(ROOT).as_posix(),source_sha256=sha256(path/'validation_metrics_identity_fp32.json'),plots_dir=(path/'figures_identity_fp32').relative_to(ROOT).as_posix(),confusion_matrix_path=(path/'figures_identity_fp32/confusion_matrix.csv').relative_to(ROOT).as_posix(),notes='Control embedded within S13 inference: same S12 checkpoints,224px,FP32,identity only; no training/test/new checkpoints. Runtime not separately timed; total S13 includes12 passes and figures.')
    control_config=dict(config,experiment_id=CONTROL,views=['identity'],question=control['method'],derived_from=ID,scope='Saved identity subset of the predeclared S13 run; not a separately launched experiment')
    write_json(path/'identity_control_config.json',control_config);control['config_path']=(path/'identity_control_config.json').relative_to(ROOT).as_posix()
    upsert(control);write_json(path/'identity_control_record.json',control)
    assert untouched==[r for r in read_registry() if r['experiment_id'] not in (ID,CONTROL)]
    assert not (ROOT/'checkpoints/structured'/ID).exists()
    report=dict(status='verified_completed',experiment_id=ID,results=results,paired_gains=gains,validation_images=1503,verified_view_tables=12,aggregate_prediction_tables=2,source_hashes=hashes,parent_checkpoint_hashes=config['checkpoint_sha256'],signature=signature,registry_entries=[ID,CONTROL],historical_rows_preserved=sum(r['era']=='legacy' for r in untouched),all_other_registry_rows_unchanged=True,runtime_seconds=record['runtime_seconds'],gpu_used_for_closeout=False,test_images_loaded=False,decision='Reject four-flip TTA; retain S12 architecture/ensemble reference. FP32 control adds one image, not a meaningful methodology gain.')
    write_json(path/'closeout_verification.json',report);write_json(OUT/'closeout_verification.json',report)
    print(json.dumps(dict(metrics={n:{k:m[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')} for n,m in results.items()},gains=gains,runtime_seconds=record['runtime_seconds']),indent=2))
if __name__=='__main__':main()
