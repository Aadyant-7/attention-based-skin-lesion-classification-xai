"""Checkpoint-derived bookkeeping and one fixed FP32 ensemble closeout."""
import json
import numpy as np
import pandas as pd
from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json
from research.registry import upsert
from research.plots import validate_metrics, metric_figures, training_figures, comparison_figures
from research.strict_train import atomic_checkpoint
from research.short_screening.lesion_bag_screen import report

COLS=[f'p_{c}' for c in CLASSES]


def record(c,payload,status):
    row=dict(experiment_id=c['experiment_id'],era='structured',record_kind='trained_backbone',
        phase='final_cbam_development',protocol=c['protocol'],evaluation_split='validation',
        split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],model=c['model'],
        pretrained_weights=c['weights'],attention=c['attention'],method=c['question'],seed=c['seed'],
        image_size=c['image_size'],preprocessing=c['preprocessing'],augmentation=c['augmentation'],
        imbalance=c['imbalance'],loss=c['loss'],optimizer=c['optimizer'],batch_size=c['batch_size'],
        backbone_lr=c['learning_rate']['backbone'],head_lr=c['learning_rate']['head_attention'],
        config_path=c['results']+'/config.json',epochs=payload['epoch'],status=status,
        history_path=c['results']+'/history.csv',checkpoint=c['checkpoints']+'/best.pt',
        notes='Post-test exploratory development; no original-test evaluation; combined recipe, not pure CBAM ablation')
    if payload['best_accuracy']:
        best=payload['best_accuracy'];row.update(best_epoch=best['epoch'],**{k:best['metrics'][k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']},
            val_loss=best['metrics']['loss'],metrics_path=c['results']+'/validation_metrics.json',plots_dir=c['results']+'/figures',
            checkpoint_available_local=True,checkpoint_sha256=sha256(ROOT/c['checkpoints']/'best.pt'),runtime_seconds=payload['runtime_seconds'])
    return row


def package(folder,metrics,predictions,figures=False,title='S79 exploratory validation'):
    validate_metrics(metrics)
    if len(predictions)!=sum(v['support'] for v in metrics['per_class'].values()):raise ValueError('Prediction/support mismatch')
    frame=pd.DataFrame(predictions);p=frame[COLS].to_numpy();y=frame.true_class.map(dict(zip(CLASSES,range(7)))).to_numpy()
    if report(y,p)['confusion_matrix']!=metrics['confusion_matrix']:raise ValueError('Predictions and metrics disagree')
    write_json(folder/'validation_metrics.json',metrics);write_csv(folder/'validation_predictions.csv',predictions)
    write_csv(folder/'validation_probabilities.csv',frame[['image_id']+COLS].to_dict('records'))
    write_csv(folder/'per_class_metrics.csv',[dict(class_name=k,**v) for k,v in metrics['per_class'].items()])
    if figures:metric_figures(metrics,folder/'figures',title)


def repair(payload,folder,ck,c,publish=True):
    """Safe after latest.pt commit; never performs an optimizer update."""
    if payload['history']:write_csv(folder/'history.csv',payload['history'])
    for key,name in [('best_accuracy','best.pt'),('best_macro_f1','best_macro_f1.pt')]:
        best=payload[key]
        if best is None:continue
        atomic_checkpoint(ck/name,dict(config=c,signature=payload['signature'],class_order=list(CLASSES),
            selection_metric='accuracy' if key=='best_accuracy' else 'macro_f1',best_epoch=best['epoch'],
            metrics=best['metrics'],model=best['model']))
        target=folder if key=='best_accuracy' else folder/'macro_f1_selected'
        package(target,best['metrics'],best['predictions'])
    if payload.get('last_validation'):
        last=payload['last_validation'];package(folder/'final_epoch',last['metrics'],last['predictions'])
    status='initialized' if payload['epoch']==0 else 'running'
    row=record(c,payload,status);write_json(folder/'record.json',row)
    if publish:upsert(row)
    write_json(folder/'progress.json',dict(status=status,committed_epoch=payload['epoch'],
        meaningful_stopping=payload['meaningful_stopping'],training_precision='amp_fp16',validation_precision='fp32'))


def fp32_parts(signature,val):
    parts=[]
    for item,source in zip(signature['fp32_reference_caches'],signature['phase1']['ensemble']['sources']):
        assert sha256(ROOT/item['path'])==item['sha256']
        with np.load(ROOT/item['path'],allow_pickle=False) as a:
            assert a['ids'].tolist()==val.image_id.tolist() and a['y'].tolist()==val.label.tolist()
            assert str(a['checkpoint_sha256'])==source['checkpoint_sha256'] and str(a['signature_sha256'])==signature['s76_signature_sha256']
            p=a['probabilities'];assert p.dtype==np.float32 and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5);parts.append(p.copy())
    return parts


def ensemble(payload,folder,c,val,publish=True):
    best=payload['best_accuracy'];frame=pd.DataFrame(best['predictions'])
    assert frame.image_id.tolist()==val.image_id.tolist()
    parts=fp32_parts(payload['signature'],val);control=np.mean(parts,axis=0)
    m0=report(val.label.to_numpy(),control)
    assert abs(m0['accuracy']-.936127744510978)<1e-12 and abs(m0['macro_f1']-.8868574111567237)<1e-12
    p=np.mean([frame[COLS].to_numpy(dtype=np.float32)]+parts[1:],axis=0);m=report(val.label.to_numpy(),p)
    f=val[['image_id','lesion_id']].copy();f['true_class']=val.diagnosis;f['predicted_class']=[CLASSES[i] for i in p.argmax(1)];f[COLS]=p
    out=folder/'fixed_ensemble';package(out,m,f.to_dict('records'),True,'S79 replacement equal-five | exploratory validation')
    correct=control.argmax(1)==val.label.to_numpy();new=p.argmax(1)==val.label.to_numpy()
    gain=int((~correct&new).sum());lost=int((correct&~new).sum());net=gain-lost
    gate=c['material_gate'];passed=net>=gate['minimum_net_correct'] and m['accuracy']-m0['accuracy']>=gate['minimum_accuracy_gain']-1e-12 and m['macro_f1']>=m0['macro_f1'] and m['per_class']['mel']['recall']>=m0['per_class']['mel']['recall']
    write_json(out/'summary.json',dict(status='completed',accuracy=m['accuracy'],macro_f1=m['macro_f1'],reference_metrics=m0,
        gained=gain,lost=lost,net_correct=net,material_gate_passed=bool(passed),weights=[.2]*5,precision='FP32',views=['identity'],
        candidate_epoch=best['epoch'],checkpoint_selection='earliest standalone accuracy maximum',
        candidate_checkpoint_sha256=sha256(ROOT/c['checkpoints']/'best.pt'),test_loaded=False,
        caveat='Previously reused exploratory validation; no independent test claim; no rescue search'))
    comparison_figures([dict(display_name='Fresh FP32 S53 reference',accuracy=m0['accuracy'],macro_f1=m0['macro_f1']),
        dict(display_name='S79 CBAM replacement equal-five',accuracy=m['accuracy'],macro_f1=m['macro_f1'])],out/'comparison_figures','Fixed attention-inclusive ensemble comparison')
    if publish:
        upsert(dict(experiment_id='s80_s79_replacement_equal_five_exploratory_seed42',era='structured',record_kind='fixed_probability_fusion',
            phase='final_cbam_development',protocol=c['protocol'],evaluation_split='validation',split_manifest=c['split_manifest'],split_sha256=c['split_sha256'],
            method=c['fusion_rule'],model='convnext_tiny_cbam+convnext_small+densenet201+efficientnet_v2_s+efficientnet_b0',
            ensemble_weights='[0.2,0.2,0.2,0.2,0.2]',seed=42,image_size=224,epochs=0,status='completed',
            decision='material_gate_passed' if passed else 'no_material_gain',checkpoint=c['checkpoints']+'/best.pt',
            metrics_path=relative(out/'validation_metrics.json'),plots_dir=relative(out/'figures'),config_path=c['results']+'/config.json',
            notes='One fixed FP32 identity ensemble; no epoch/weight/member search or original-test inference',
            **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1']}))
    return bool(passed)


def closeout(payload,folder,ck,c,val,publish=True):
    repair(payload,folder,ck,c,publish=False)
    for key,target in [('best_accuracy',folder),('best_macro_f1',folder/'macro_f1_selected')]:
        b=payload[key];package(target,b['metrics'],b['predictions'],True)
    last=payload['last_validation'];package(folder/'final_epoch',last['metrics'],last['predictions'],True)
    training_figures(pd.DataFrame(payload['history']),folder/'figures','S79 bounded training | exploratory validation',selection_metric='accuracy')
    passed=ensemble(payload,folder,c,val,publish)
    reason=payload['meaningful_stopping']['reason']
    if reason=='running':raise ValueError('Cannot close out an unfinished run')
    write_json(folder/'training_summary.json',dict(status='completed',epochs=payload['epoch'],stop_reason=reason,
        best_accuracy_epoch=payload['best_accuracy']['epoch'],best_macro_f1_epoch=payload['best_macro_f1']['epoch'],
        best_accuracy=payload['best_accuracy']['metrics']['accuracy'],best_macro_f1=payload['best_macro_f1']['metrics']['macro_f1'],
        final_metrics=last['metrics'],meaningful_stopping=payload['meaningful_stopping'],optimizer_updates=payload['optimizer_updates'],
        runtime_seconds=payload['runtime_seconds'],fixed_ensemble_material_gate_passed=passed,test_loaded=False))
    row=record(c,payload,'completed');row['decision']='completed_bounded_recipe';write_json(folder/'record.json',row)
    if publish:upsert(row)
    write_json(folder/'progress.json',dict(status='completed',committed_epoch=payload['epoch'],stop_reason=reason,active_process=False))
