"""CPU-only verification/closeout of the completed first structured control."""
import json
import numpy as np
import pandas as pd
import torch
from research.common import ROOT, CLASSES, sha256, write_json, write_csv
from research.plots import validate_metrics, comparison_figures
from research.registry import read_registry, upsert

ID='s01_efficientnet_b0_none_strict_seed42'


def main():
    path=ROOT/'results/structured_experiments'/ID
    config=json.loads((path/'config.json').read_text())
    record=json.loads((path/'record.json').read_text())
    metrics=json.loads((path/'validation_metrics.json').read_text())
    cm=validate_metrics(metrics)
    history=pd.read_csv(path/'history.csv')
    predictions=pd.read_csv(path/'validation_predictions.csv')
    manifest=pd.read_csv(ROOT/config['split_manifest'])
    assert sha256(ROOT/config['split_manifest'])==config['split_sha256']
    val=manifest.query("split=='val'").set_index('image_id')
    assert len(predictions)==len(val)==1503 and predictions.image_id.is_unique
    found=predictions.set_index('image_id').loc[val.index]
    assert found.true_class.equals(val.diagnosis)
    probabilities=found[[f'p_{c}' for c in CLASSES]].to_numpy()
    assert np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
    assert np.allclose(probabilities.sum(1),1,atol=1e-6)
    predicted=np.asarray(CLASSES)[probabilities.argmax(1)]
    assert np.array_equal(predicted,found.predicted_class)
    from sklearn.metrics import confusion_matrix
    assert np.array_equal(cm,confusion_matrix(val.diagnosis,predicted,labels=list(CLASSES)))
    assert history.epoch.tolist()==list(range(1,21)) and record['epochs']==20
    best_epoch=int(history.loc[history.val_macro_f1.idxmax(),'epoch'])
    assert best_epoch==record['best_epoch']==18
    best_row=history.loc[history.epoch==best_epoch].iloc[0]
    for k in ('accuracy','macro_precision','macro_recall','macro_f1'):
        assert np.isclose(metrics[k],best_row['val_'+k],rtol=0,atol=1e-10)
        assert np.isclose(record[k],metrics[k],rtol=0,atol=1e-10)
    best=ROOT/record['checkpoint'];latest=best.parent/'latest.pt'
    # Own local full-state checkpoints; CPU only, no model construction/inference.
    selected=torch.load(best,map_location='cpu',weights_only=False)
    final=torch.load(latest,map_location='cpu',weights_only=False)
    assert selected['best_epoch']==best_epoch and selected['metrics']==metrics and selected['config']==config
    pd.testing.assert_frame_equal(pd.DataFrame(final['history']),history,check_exact=False,rtol=1e-12,atol=1e-12)
    assert final['best']['epoch']==best_epoch and final['best']['metrics']==metrics
    assert sha256(best)==record['checkpoint_sha256']
    assert all(torch.equal(selected['model'][k],v) for k,v in final['best']['model'].items())
    figures=path/'figures'
    names=('accuracy_curves','loss_curves','macro_f1_curve','confusion_matrix',
           'confusion_matrix_normalized','per_class_metrics','class_support')
    for name in names:
        assert (figures/(name+'.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert (figures/(name+'.pdf')).read_bytes().startswith(b'%PDF')
    matrix=pd.read_csv(figures/'confusion_matrix.csv').set_index('true_class').loc[list(CLASSES),list(CLASSES)].to_numpy()
    assert np.array_equal(matrix,cm)
    normalized=pd.read_csv(figures/'confusion_matrix_normalized.csv').set_index('true_class').loc[list(CLASSES),list(CLASSES)].to_numpy()
    assert np.allclose(normalized,cm/cm.sum(1,keepdims=True))
    classes=pd.read_csv(figures/'per_class_metrics.csv').set_index('class')
    for c in CLASSES:
        for k,v in metrics['per_class'][c].items(): assert np.isclose(classes.loc[c,k],v)
    historical_path=ROOT/'results/runs/efficientnet_b0_cbam_weighted_v1/validation_metrics.json'
    old=json.loads(historical_path.read_text());validate_metrics(old)
    assert [old['per_class'][c]['support'] for c in CLASSES]==cm.sum(1).tolist()
    baseline=read_registry();legacy=[r for r in baseline if r['era']=='legacy']
    record.update(status='completed',decision='strict_reference_retained_screening_moves_to_exploratory',
                  source_availability='original',notes=record['notes'].split(' Closed out on CPU:')[0]+' Closed out on CPU: checkpoint/history/predictions/matrix/class scores verified. Historical comparison is descriptive, not an isolated CBAM ablation.')
    upsert(record);write_json(path/'record.json',record)
    assert legacy==[r for r in read_registry() if r['era']=='legacy']
    out=ROOT/'results/model_comparison/structured/s01_vs_legacy_b0'
    rows=[dict(display_name='Structured B0 | GAP head | batch16/acc2',experiment_id=ID,
               accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],best_epoch=18,comparison='same strict cohort; different recipes'),
          dict(display_name='Historical B0+CBAM | MLP head | batch64',experiment_id='efficientnet_b0_cbam_weighted_v1',
               accuracy=old['accuracy'],macro_f1=old['macro_f1'],best_epoch=16,comparison='descriptive; not CBAM ablation')]
    comparison_figures(rows,out,'Same strict validation cohort | different recipes\nDescriptive comparison; added-CBAM effect not isolated')
    write_csv(out/'per_class_comparison.csv',[dict(class_name=c,support=metrics['per_class'][c]['support'],
               **{f'new_{k}':metrics['per_class'][c][k] for k in ('precision','recall','f1')},
               **{f'historical_{k}':old['per_class'][c][k] for k in ('precision','recall','f1')},
               f1_delta=metrics['per_class'][c]['f1']-old['per_class'][c]['f1']) for c in CLASSES])
    report=dict(status='verified_completed',experiment_id=ID,epochs=20,best_epoch=18,selected_metrics=metrics,
       peak_accuracy=float(history.val_accuracy.max()),peak_accuracy_epoch=int(history.loc[history.val_accuracy.idxmax(),'epoch']),
       peak_accuracy_checkpoint_available=False,runtime_seconds=record['runtime_seconds'],
       train_accuracy_selected=float(best_row.train_accuracy),train_accuracy_final=float(history.iloc[-1].train_accuracy),
       minimum_val_loss_epoch=int(history.loc[history.val_loss.idxmin(),'epoch']),
       accuracy_delta_percentage_points=(metrics['accuracy']-old['accuracy'])*100,
       macro_f1_delta=metrics['macro_f1']-old['macro_f1'],
       selected_melanoma_recall=metrics['per_class']['mel']['recall'],historical_melanoma_recall=old['per_class']['mel']['recall'],
       historical_comparison_scope='Same strict manifest/cohort/ImageNet/224/flip/sqrt class weights; changed head/attention/microbatch/effective batch/loss aggregation. No causal CBAM or significance claim.',
       verified_checks=['checkpoint states','20-epoch history','earliest maximum macro-F1','all1503 prediction identities/labels/probabilities','matrix and class scores','7 PNG/PDF figure pairs','473 historical registry rows unchanged'],
       source_hashes={n:sha256(path/n) for n in ('config.json','history.csv','validation_metrics.json','validation_predictions.csv')},
       best_checkpoint_sha256=sha256(best),latest_checkpoint_sha256=sha256(latest),
       historical_metrics_sha256=sha256(historical_path),gpu_used=False,test_images_opened=False)
    write_json(path/'closeout_verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected_metrics','source_hashes')},indent=2))


if __name__=='__main__': main()
