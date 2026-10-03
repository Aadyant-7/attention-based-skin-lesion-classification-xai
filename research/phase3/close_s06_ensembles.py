"""Verify the three predeclared S06 fusions and preserve every outcome; CPU only."""
import json
import numpy as np
import pandas as pd
from research.common import ROOT, CLASSES, sha256, write_json, write_csv
from research.fuse import preflight
from research.plots import comparison_figures
from research.phase3.close_s02 import load, verify_predictions, verify_figures
from research.phase3.close_s06 import PEERS
from research.registry import read_registry, upsert

OUT = ROOT / 'results/model_comparison/structured/s06_bounded_ensembles'


def main():
    declared = load(OUT/'predeclared_candidates.json')
    baseline = read_registry()
    candidate_ids = {c['experiment_id'] for c in declared['candidates']}
    untouched = [r for r in baseline if r['experiment_id'] not in candidate_ids]
    rows, classes, verified = [], [], {}
    references = {}
    for label, rid in PEERS.items():
        path = ROOT/'results/structured_experiments'/rid
        r, m = load(path/'record.json'), load(path/'validation_metrics.json')
        rows.append(dict(display_name=label, experiment_id=rid, accuracy=m['accuracy'], macro_precision=m['macro_precision'],
            macro_recall=m['macro_recall'], macro_f1=m['macro_f1'], melanoma_recall=m['per_class']['mel']['recall'],
            inference_model_passes=2 if rid.startswith('s04') else 1, scope='exploratory validation; accuracy winners; cost differs'))
        classes.extend(dict(experiment_id=rid, class_name=c, **m['per_class'][c]) for c in CLASSES)
        if rid.startswith(('s04','s06')):
            references[rid] = (m, pd.read_csv(path/'validation_predictions.csv').set_index('image_id'))

    for entry in declared['candidates']:
        assert sha256(ROOT/entry['config_path']) == entry['config_sha256']
        rid = entry['experiment_id']
        path = ROOT/'results/structured_experiments'/rid
        config, record, metrics = load(path/'config.json'), load(path/'record.json'), load(path/'validation_metrics.json')
        assert config == load(ROOT/entry['config_path']) and record['status'] == 'completed'
        frame, val, parents, probabilities = preflight(config)
        cm = verify_predictions(path/'validation_predictions.csv', metrics, val)
        found = pd.read_csv(path/'validation_predictions.csv').set_index('image_id').loc[val.index]
        fused = sum(w*p for w,p in zip(config['weights'], probabilities))
        assert np.allclose(found[[f'p_{c}' for c in CLASSES]], fused, rtol=0, atol=1e-12)
        labels = val.label.to_numpy()
        counts = frame.query("split=='train'").diagnosis.value_counts()
        weights = np.sqrt(len(frame.query("split=='train'"))/np.asarray([counts[c] for c in CLASSES]))
        weights /= weights.mean()
        loss = float(np.average(-np.log(np.clip(fused[np.arange(len(labels)),labels],1e-12,1)), weights=weights[labels]))
        assert np.isclose(metrics['loss'], loss, rtol=0, atol=1e-12)
        assert record['source_sha256'] == sha256(path/'validation_metrics.json')
        for field in ('accuracy','macro_precision','macro_recall','macro_f1'):
            assert np.isclose(record[field], metrics[field], rtol=0, atol=1e-12)
        assert not record['epochs'] and not record['best_epoch']
        assert not (ROOT/'checkpoints/structured'/rid).exists()
        verify_figures(path/'figures', cm, metrics)
        gains = {}
        ok = found.predicted_class.to_numpy() == val.diagnosis.to_numpy()
        for reference, (m, predictions) in references.items():
            other_ok = predictions.loc[val.index].predicted_class.to_numpy() == val.diagnosis.to_numpy()
            gains[reference] = dict(fixed=int((ok & ~other_ok).sum()), broken=int((~ok & other_ok).sum()),
                both_wrong=int((~ok & ~other_ok).sum()), net_additional_correct=int(ok.sum()-other_ok.sum()),
                accuracy_gain_percentage_points=100*(metrics['accuracy']-m['accuracy']),
                macro_f1_gain=metrics['macro_f1']-m['macro_f1'])
        record.update(decision='preserve_bounded_result_no_accuracy_gain_over_s06',
            notes=record['notes'].split(' Closed out on CPU:')[0]+' Closed out on CPU: exact fixed probabilities, loss, labels, matrix/class metrics and figures verified; three candidates disclosed. No accuracy gain over S06; preserve balanced trade-offs and inference cost.')
        upsert(record)
        write_json(path/'record.json', record)
        report = dict(status='verified_completed', experiment_id=rid, metrics=metrics, validation_images=len(val),
            correct=int(ok.sum()), paired_gains=gains, config_sha256=entry['config_sha256'],
            source_hashes={n:sha256(path/n) for n in ('config.json','validation_metrics.json','validation_predictions.csv')},
            parent_prediction_hashes=config['prediction_sha256'], parent_checkpoint_identities=config['checkpoint_sha256'],
            verified_checks=['exact equal probabilities','all validation IDs and labels','weighted loss','matrix and class scores','four PNG/PDF pairs and CSVs'],
            gpu_used=False, test_images_loaded=False, new_checkpoint=False, selection_scope='three predeclared candidates; validation-selected parents')
        write_json(path/'closeout_verification.json', report)
        verified[rid] = dict(metrics=metrics, paired_gains=gains, inference_model_passes=len(parents))
        display = rid.split('_',1)[0].upper()+' equal '+ '/'.join(p.split('_')[0].upper() for p in config['parent_run_ids'])
        rows.append(dict(display_name=display, experiment_id=rid, accuracy=metrics['accuracy'], macro_precision=metrics['macro_precision'],
            macro_recall=metrics['macro_recall'], macro_f1=metrics['macro_f1'], melanoma_recall=metrics['per_class']['mel']['recall'],
            inference_model_passes=len(parents), scope='one of three predeclared equal candidates; accuracy-selected parents; exploratory validation'))
        classes.extend(dict(experiment_id=rid, class_name=c, **metrics['per_class'][c]) for c in CLASSES)

    assert untouched == [r for r in read_registry() if r['experiment_id'] not in candidate_ids]
    comparison_figures(rows, OUT, 'Same exploratory validation | all three fixed S06 fusion candidates\nNo weight search; 1/2/3 model-pass costs must be reported')
    write_csv(OUT/'per_class_comparison.csv', classes)
    write_json(OUT/'closeout_verification.json', dict(status='all_three_verified', results=verified,
        all_other_registry_rows_unchanged=True, historical_rows_preserved=sum(r['era']=='legacy' for r in untouched),
        decision='S06 remains accuracy and cost reference; S09 is a balanced fusion alternative without accuracy gain. Propose one stronger distinct pretrained package before further fusion.',
        gpu_used=False, test_images_loaded=False))
    print(json.dumps({rid:dict(accuracy=v['metrics']['accuracy'], macro_f1=v['metrics']['macro_f1'], paired_gains=v['paired_gains']) for rid,v in verified.items()}, indent=2))


if __name__ == '__main__':
    main()
