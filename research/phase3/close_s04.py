"""Verify fixed S04 fusion and paired exploratory improvements; saved arrays only."""
import json
import numpy as np
import pandas as pd
from research.common import ROOT, CLASSES, sha256, write_json, write_csv
from research.fuse import preflight
from research.phase3.close_s02 import load, verify_predictions, verify_figures
from research.plots import comparison_figures, validate_metrics
from research.registry import read_registry, upsert

ID = 's04_s02_s03_equal_probability_exploratory_seed42'


def main():
    path = ROOT / 'results/structured_experiments' / ID
    config, record = load(path/'config.json'), load(path/'record.json')
    assert record['status'] == 'completed' and record['record_kind'] == 'fixed_probability_fusion'
    frame, val, parents, probabilities = preflight(config)
    metrics = load(path/'validation_metrics.json')
    cm = verify_predictions(path/'validation_predictions.csv', metrics, val)
    found = pd.read_csv(path/'validation_predictions.csv').set_index('image_id').loc[val.index]
    fused = sum(w*p for w,p in zip(config['weights'],probabilities))
    assert np.allclose(found[[f'p_{c}' for c in CLASSES]], fused, rtol=0, atol=1e-12)
    labels = val.diagnosis.map(dict(zip(CLASSES,range(7)))).to_numpy()
    counts = frame.query("split=='train'").diagnosis.value_counts()
    weights = np.sqrt(len(frame.query("split=='train'"))/np.asarray([counts[c] for c in CLASSES]));weights/=weights.mean()
    loss = float(np.average(-np.log(np.clip(fused[np.arange(len(labels)),labels],1e-12,1)),weights=weights[labels]))
    assert np.isclose(metrics['loss'],loss,rtol=0,atol=1e-12)
    for key in ('accuracy','macro_precision','macro_recall','macro_f1'):
        assert np.isclose(record[key],metrics[key],rtol=0,atol=1e-12)
    assert record['source_sha256']==sha256(path/'validation_metrics.json')
    assert not record['epochs'] and not record['best_epoch']
    assert not (ROOT/'checkpoints/structured'/ID).exists()
    verify_figures(path/'figures',cm,metrics)
    baseline=read_registry();other_rows=[r for r in baseline if r['experiment_id']!=ID]
    record.update(decision='retain_fixed_fusion_reference_propose_matched_b0_cbam',
        notes=record['notes'].split(' Closed out on CPU:')[0]+' Closed out on CPU: exact fixed probability mean, all 1503 labels/predictions, loss/matrix/class metrics/figures verified. Accuracy and macro-F1 improve over both parents; akiec/vasc remain trade-offs.')
    upsert(record);write_json(path/'record.json',record)
    assert other_rows==[r for r in read_registry() if r['experiment_id']!=ID]
    out=ROOT/'results/model_comparison/structured/s04_fixed_fusion'
    fusion_ok=found.predicted_class.to_numpy()==val.diagnosis.to_numpy()
    gains={};rows=[];class_rows=[]
    for parent,m in parents:
        name=parent['experiment_id']
        assert np.array_equal(validate_metrics(m).sum(1),cm.sum(1))
        parent_predictions=pd.read_csv(ROOT/'results/structured_experiments'/name/'validation_predictions.csv').set_index('image_id').loc[val.index]
        parent_ok=parent_predictions.predicted_class.to_numpy()==val.diagnosis.to_numpy()
        gains[name]=dict(correct=int(parent_ok.sum()),fixed=int((~parent_ok&fusion_ok).sum()),
            broken=int((parent_ok&~fusion_ok).sum()),both_correct=int((parent_ok&fusion_ok).sum()),
            both_wrong=int((~parent_ok&~fusion_ok).sum()),net_additional_correct=int(fusion_ok.sum()-parent_ok.sum()),
            accuracy_gain_percentage_points=(metrics['accuracy']-m['accuracy'])*100,
            macro_f1_gain=metrics['macro_f1']-m['macro_f1'],
            macro_recall_gain=metrics['macro_recall']-m['macro_recall'])
        rows.append(dict(display_name=parent['model']+' | single model',experiment_id=name,accuracy=m['accuracy'],
            macro_precision=m['macro_precision'],macro_recall=m['macro_recall'],macro_f1=m['macro_f1'],
            melanoma_recall=m['per_class']['mel']['recall'],protocol=config['protocol'],scope='matched parent; accuracy-selected'))
    rows.append(dict(display_name='S04 fixed 50/50 fusion | two model passes',experiment_id=ID,accuracy=metrics['accuracy'],
        macro_precision=metrics['macro_precision'],macro_recall=metrics['macro_recall'],macro_f1=metrics['macro_f1'],
        melanoma_recall=metrics['per_class']['mel']['recall'],protocol=config['protocol'],scope='one predeclared fixed fusion; same validation cohort'))
    comparison_figures(rows,out,'Same exploratory validation | seed42 parents\nOne fixed fusion; no weight search; two-model inference cost')
    for c in CLASSES:
        class_rows.append(dict(class_name=c,support=metrics['per_class'][c]['support'],
            **{f's02_{k}':parents[0][1]['per_class'][c][k] for k in ('precision','recall','f1')},
            **{f's03_{k}':parents[1][1]['per_class'][c][k] for k in ('precision','recall','f1')},
            **{f's04_{k}':metrics['per_class'][c][k] for k in ('precision','recall','f1')}))
    write_csv(out/'per_class_comparison.csv',class_rows)
    write_json(out/'paired_gains.json',gains)
    for label,filename,selector in (
        ('Historical weighted B0+CBAM single model','results/exploratory/image_level_weighted_b0_cbam_v1_summary.json','best_validation'),
        ('Historical B0 four-stream ensemble','results/exploratory/multires_ensemble/four_equal.json',None),
        ('Historical B0+PanDerm eight-stream ensemble','results/exploratory/panderm_base_image_level_tta_v1/b0_four_plus_four_svc_views_eight_equal.json',None),
    ):
        old=next(r for r in baseline if r['metrics_path']==filename)
        assert old['protocol']==config['protocol'] and old['split_sha256']==config['split_sha256']
        m=load(ROOT/filename);m=m[selector] if selector else m
        assert np.array_equal(validate_metrics(m).sum(1),cm.sum(1))
        rows.append(dict(display_name=label,experiment_id=old['experiment_id'],accuracy=m['accuracy'],macro_precision=m['macro_precision'],
            macro_recall=m['macro_recall'],macro_f1=m['macro_f1'],melanoma_recall=m['per_class']['mel']['recall'],
            protocol=config['protocol'],scope='same exploratory cohort; historical recipe/selection/cost differ'))
    comparison_figures(rows,ROOT/'results/model_comparison/structured/s04_exploratory_context',
        'Same exploratory cohort | historical recipes/costs differ\nStrict results excluded; historical maxima validation-selected')
    report=dict(status='verified_completed',experiment_id=ID,protocol=config['protocol'],metrics=metrics,
        correct=int(np.trace(cm)),validation_images=int(cm.sum()),paired_gains=gains,
        evaluation_seconds=record['runtime_seconds'],source_hashes={name:sha256(path/name) for name in ('config.json','validation_metrics.json','validation_predictions.csv')},
        parent_prediction_hashes=config['prediction_sha256'],parent_checkpoint_identities=config['checkpoint_sha256'],
        verified_checks=['exact 50/50 probabilities','all 1503 IDs/labels/predictions','weighted loss','matrix/macro/class metrics','four PNG/PDF pairs and CSVs'],
        historical_rows_preserved=sum(r['era']=='legacy' for r in other_rows),all_other_registry_rows_unchanged=True,
        scope='Observed exploratory validation improvement; parent checkpoints validation-selected; not lesion-independent/test/significance evidence',
        gpu_used=False,test_images_loaded=False,new_training=False,new_checkpoint=False)
    write_json(path/'closeout_verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','source_hashes')},indent=2))


if __name__=='__main__':main()
