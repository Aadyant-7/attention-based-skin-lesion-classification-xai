"""Only fixed strict validation closeout; never creates a test loader."""
import json
import numpy as np
import pandas as pd
import torch
from .common import ROOT,CLASSES,sha256,write_csv,write_json,atomic_text,relative
from .strict_protocol import OUT,partition_metadata
from .strict_train import metric_report,code_hashes,tensor_nonfinite_names
from .plots import validate_metrics,metric_figures,comparison_figures,training_figures,save
from .registry import FIELDS,upsert

def load_predictions(path):
    frame=pd.read_csv(path)
    dev,_=partition_metadata()
    val=dev.loc[dev.split=='val'].set_index('image_id')
    if frame.image_id.duplicated().any() or set(frame.image_id)!=set(val.index):
        raise ValueError('Strict validation identities mismatch')
    frame=frame.set_index('image_id').loc[val.index].reset_index()
    if not np.array_equal(frame.true_class,val.diagnosis): raise ValueError('Validation labels mismatch')
    p=frame[[f'p_{c}' for c in CLASSES]].to_numpy(float)
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Invalid probability vectors')
    return frame,val.label.to_numpy(),p

def verify_model(config):
    rid=config['experiment_id'];path=ROOT/'results/structured_experiments'/rid
    ck=ROOT/'checkpoints/structured'/rid
    record=json.loads((path/'record.json').read_text())
    if record['status']!='completed' or record['protocol']!='strict_lesion_disjoint': raise ValueError('Model incomplete')
    if json.loads((path/'config.json').read_text())!=config: raise ValueError('Config changed')
    pre=json.loads((path/'pretraining.json').read_text())
    if pre['weights']!=config['weights'] or not pre['fresh_imagenet_initialization'] or pre['exploratory_checkpoint_reused']:
        raise ValueError('Invalid initialization provenance')
    history=pd.read_csv(path/'history.csv')
    if history.epoch.tolist()!=list(range(1,len(history)+1)) or len(history)>50 or not (history.validation_precision=='fp32').all():
        raise ValueError('Invalid strict history')
    manifest={}
    for name,suffix in [('best.pt',''),('best_macro_f1.pt','_macro_f1'),('latest.pt','_latest')]:
        checkpoint=torch.load(ck/name,map_location='cpu',weights_only=False)
        if checkpoint['config']!=config or tensor_nonfinite_names(checkpoint['model']): raise ValueError('Invalid checkpoint')
        if 'code_hashes' in checkpoint and checkpoint['code_hashes']!=code_hashes(): raise ValueError('Checkpoint source mismatch')
        if name=='latest.pt' and (len(checkpoint['history'])!=len(history) or tensor_nonfinite_names(checkpoint['optimizer'])):
            raise ValueError('Invalid committed boundary')
        frame,y,p=load_predictions(path/f'validation_predictions{suffix}.csv')
        m=json.loads((path/f'validation_metrics{suffix}.json').read_text());validate_metrics(m)
        recomputed=metric_report(y,p,m['loss'])
        for key in ['accuracy','macro_precision','macro_recall','macro_f1']:
            if abs(m[key]-recomputed[key])>1e-8: raise ValueError('Prediction metrics mismatch')
        if name=='best.pt':
            epoch=int(history.loc[history.val_accuracy.idxmax(),'epoch'])
            if checkpoint['best_epoch']!=epoch or abs(m['accuracy']-history.val_accuracy.max())>1e-8: raise ValueError('Accuracy winner mismatch')
        if name=='best_macro_f1.pt':
            epoch=int(history.loc[history.val_macro_f1.idxmax(),'epoch'])
            if checkpoint['best_epoch']!=epoch or abs(m['macro_f1']-history.val_macro_f1.max())>1e-8: raise ValueError('F1 winner mismatch')
        write_csv(path/f'validation_probabilities{suffix}.csv',frame[['image_id']+[f'p_{c}' for c in CLASSES]].to_dict('records'))
        for image in ('confusion_matrix','confusion_matrix_normalized','per_class_metrics'):
            folder=path/('figures' if not suffix else 'figures'+suffix)
            for extension in ('csv','png','pdf'):
                if not (folder/(image+'.'+extension)).is_file(): raise FileNotFoundError(folder/(image+'.'+extension))
        manifest[name]=dict(sha256=sha256(ck/name),metrics_sha256=sha256(path/f'validation_metrics{suffix}.json'),prediction_sha256=sha256(path/f'validation_predictions{suffix}.csv'))
        del checkpoint
    write_json(path/'artifact_verification.json',dict(status='verified',checkpoint_manifest=manifest,test_loader=False))
    # Existing training curves contain both train and validation traces; add LR graph.
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for key in ('backbone_lr','head_lr'):ax.plot(history.epoch,history[key],label=key)
    ax.set(xlabel='Epoch',ylabel='Learning rate',yscale='log',title=f"{config['model']} | strict validation recipe")
    ax.legend();save(fig,path/'figures','lr_history')
    return json.loads((path/'validation_metrics.json').read_text()),record

def equal_probabilities(arrays):
    if len(arrays)!=3 or len({a.shape for a in arrays})!=1: raise ValueError('Exactly three aligned members required')
    result=np.mean(np.stack(arrays),axis=0)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1),1,atol=1e-5):raise ValueError('Invalid ensemble')
    return result

def closeout(configs,ensemble_id):
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];arrays=[];members=[];y=None;ids=None
    for config in configs:
        m,record=verify_model(config)
        path=ROOT/'results/structured_experiments'/config['experiment_id']
        frame,labels,p=load_predictions(path/'validation_predictions.csv')
        if y is not None and (not np.array_equal(y,labels) or not np.array_equal(ids,frame.image_id)):raise ValueError('Unaligned members')
        y=labels;ids=frame.image_id.to_numpy();arrays.append(p);members.append(config['experiment_id'])
        rows.append(dict(display_name=config['model'],experiment_id=config['experiment_id'],best_epoch=record['best_epoch'],
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    p=equal_probabilities(arrays)
    train,report=partition_metadata();train=train.loc[train.split=='train']
    counts=np.array([(train.label==i).sum() for i in range(7)]);weights=np.sqrt(len(train)/(7*counts));weights/=weights.mean()
    true_prob=p[np.arange(len(y)),y]
    if (true_prob<=0).any():raise FloatingPointError('Zero true-class ensemble probability; preserve evidence, no clamping')
    loss=float((-np.log(true_prob)*weights[y]).sum()/weights[y].sum())
    m=metric_report(y,p,loss);validate_metrics(m)
    predictions=[dict(image_id=i,true_class=CLASSES[t],predicted_class=CLASSES[int(a.argmax())],**{f'p_{c}':float(a[j]) for j,c in enumerate(CLASSES)}) for i,t,a in zip(ids,y,p)]
    folder=OUT/'ensemble';write_json(folder/'validation_metrics.json',m);write_csv(folder/'validation_predictions.csv',predictions)
    write_csv(folder/'validation_probabilities.csv',[dict(image_id=i,**{f'p_{c}':float(a[j]) for j,c in enumerate(CLASSES)}) for i,a in zip(ids,p)])
    metric_figures(m,folder/'figures','Frozen equal ensemble | strict lesion-disjoint validation')
    checkpoint_manifest=[dict(model=c['model'],weights=c['weights'],checkpoint=f"checkpoints/structured/{c['experiment_id']}/best.pt",
        sha256=sha256(ROOT/f"checkpoints/structured/{c['experiment_id']}/best.pt"),best_epoch=r['best_epoch']) for c,r in zip(configs,rows)]
    write_json(folder/'frozen_ensemble.json',dict(members=checkpoint_manifest,weights=[1/3]*3,precision='fp32',views=['identity'],
        input_size=224,selection='independent earliest accuracy winners; no strict weight/epoch fusion search',split_verification=report,test_evaluated=False))
    ensemble_row=dict(display_name='Frozen equal ensemble',experiment_id=ensemble_id,best_epoch='individual accuracy winners',**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']})
    comparison_figures(rows,OUT/'figures_standalone','Strict lesion-disjoint standalone validation')
    comparison_figures(rows+[ensemble_row],OUT/'figures_comparison','Strict lesion-disjoint validation | fixed fusion')
    exploratory=json.loads((ROOT/'results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42/validation_metrics_identity_fp32.json').read_text())
    comparison_figures([dict(display_name='Exploratory image-level',accuracy=exploratory['accuracy'],macro_f1=exploratory['macro_f1']),dict(display_name='Strict lesion-disjoint',accuracy=m['accuracy'],macro_f1=m['macro_f1'])],OUT/'figures_protocols','Different validation protocols | not interchangeable')
    differences=[dict(class_name=c,**{k+'_difference':m['per_class'][c][k]-exploratory['per_class'][c][k] for k in ['precision','recall','f1']}) for c in CLASSES]
    write_csv(OUT/'exploratory_vs_strict_class_changes.csv',differences)
    write_csv(OUT/'strict_validation_comparison.csv',rows+[ensemble_row])
    strongest=max(rows,key=lambda r:r['accuracy']);idx=rows.index(strongest)
    base_correct=arrays[idx].argmax(1)==y;ens_correct=p.argmax(1)==y
    delta=dict(accuracy_percentage_points=100*(m['accuracy']-exploratory['accuracy']),macro_f1=m['macro_f1']-exploratory['macro_f1'],
        ensemble_vs_strongest_accuracy_pp=100*(m['accuracy']-strongest['accuracy']),ensemble_vs_strongest_f1=m['macro_f1']-strongest['macro_f1'],
        correct=int(ens_correct.sum()),incorrect=int((~ens_correct).sum()),gained=int((ens_correct&~base_correct).sum()),lost=int((~ens_correct&base_correct).sum()))
    write_json(OUT/'comparison_summary.json',delta)
    record={k:'' for k in FIELDS};record.update(experiment_id=ensemble_id,era='structured',record_kind='fixed_ensemble',phase='final_strict_confirmation',protocol='strict_lesion_disjoint',
        evaluation_split='validation',split_manifest=report['split_manifest'],split_sha256=report['split_sha256'],method='Frozen equal FP32 probability ensemble',
        model='efficientnet_b0 + convnext_tiny + efficientnet_v2_s',ensemble_members=json.dumps(members),ensemble_weights=json.dumps([1/3]*3),
        image_size=224,seed=42,status='completed',decision='frozen_final_strict_validation',metrics_path=relative(folder/'validation_metrics.json'),source_sha256=sha256(folder/'validation_metrics.json'),
        plots_dir=relative(folder/'figures'),confusion_matrix_path=relative(folder/'figures/confusion_matrix.csv'),config_path=relative(folder/'frozen_ensemble.json'),
        notes='Strict validation only; independent accuracy winners; locked test not loaded.',val_loss=loss,**{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']})
    write_json(folder/'record.json',record);upsert(record)
    table='| Model | Best epoch | Accuracy | Macro precision | Macro recall | Macro-F1 |\n|---|---|---:|---:|---:|---:|\n'
    for r in rows+[ensemble_row]:table+=f"| {r['display_name']} | {r['best_epoch']} | {100*r['accuracy']:.4f}% | {r['macro_precision']:.6f} | {r['macro_recall']:.6f} | {r['macro_f1']:.6f} |\n"
    classes='| Class | Precision | Recall | F1 | Support |\n|---|---:|---:|---:|---:|\n'
    for c in CLASSES:
        a=m['per_class'][c];classes+=f"| {c} | {a['precision']:.6f} | {a['recall']:.6f} | {a['f1']:.6f} | {a['support']} |\n"
    text=f"""# Final strict lesion-disjoint validation results

Architecture permanently closed: B0 + ConvNeXt-Tiny + EfficientNetV2-S, equal 1/3 averaging, FP32 identity, 224 pixels. Fresh ImageNet initialization. Locked test NOT evaluated.

{table}
## Final ensemble class performance

{classes}
Correct: {delta['correct']}; incorrect: {delta['incorrect']}. Melanoma recall {m['per_class']['mel']['recall']:.6f}; akiec recall {m['per_class']['akiec']['recall']:.6f}.

Strongest standalone: {strongest['display_name']}. Ensemble accuracy change {delta['ensemble_vs_strongest_accuracy_pp']:+.4f} percentage points; macro-F1 change {delta['ensemble_vs_strongest_f1']:+.6f}; predictions gained/lost {delta['gained']}/{delta['lost']}.

## Exploratory versus strict

Exploratory reference: {100*exploratory['accuracy']:.4f}% / {exploratory['macro_f1']:.6f}. Strict: {100*m['accuracy']:.4f}% / {m['macro_f1']:.6f}. Accuracy difference {delta['accuracy_percentage_points']:+.4f} percentage points; macro-F1 difference {delta['macro_f1']:+.6f}.

Different protocols are not interchangeable: exploratory train/validation share 563 lesions affecting 596 validation images; strict partitions share zero lesions. Validation cohorts and training duration/numerical recipe also differ. This gap cannot be attributed solely to leakage, and lower strict performance is not evidence of a failed experiment. Repeated exploratory selection adds optimism; this remains validation evidence, not test performance.

Class-wise changes: `results/final_strict/v1/exploratory_vs_strict_class_changes.csv`. Raw confusion matrix (rows true, columns predicted; {list(CLASSES)}):

```text
{np.array(m['confusion_matrix'])}
```

Saved results: `results/final_strict/v1/`; individual runs `results/structured_experiments/s28_*`, `s29_*`, `s30_*`; checkpoints `checkpoints/structured/` under matching IDs. All accuracy/F1/latest states retained. Ensemble checkpoint identities are in `ensemble/frozen_ensemble.json`. Exploratory table remains `research/MODEL_VS_ACCURACY.md`; strict table is separate.

STOP: review strict validation before one authorized locked-test evaluation, then Grad-CAM/XAI. No further architecture, weight, TTA or resolution search.
"""
    atomic_text(ROOT/'research/FINAL_STRICT_VALIDATION_RESULTS.md',text)
    write_json(OUT/'closeout_verification.json',dict(status='completed',member_count=3,fixed_weights=[1/3]*3,test_labels_read=False,test_loader=False,architecture_search_closed=True))
    return delta
