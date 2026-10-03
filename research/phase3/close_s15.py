"""CPU closeout of S06 and aligned, same-protocol error analysis."""
import json
import numpy as np
import pandas as pd
import torch
from research.common import ROOT, CLASSES, sha256, write_json, write_csv
from research.plots import metric_figures, comparison_figures
from research.registry import read_registry, upsert
from research.train import validate_config
from research.phase3.close_s02 import load, verify_predictions, verify_figures, verify_launch_code

ID = 's15_densenet201_none_exploratory_seed42'
PEERS = {
 'S06 ConvNeXt-Tiny':'s06_convnext_tiny_none_exploratory_seed42',
 'S10 EfficientNetV2-S':'s10_efficientnet_v2_s_none_exploratory_seed42',
 'S12 equal triple':'s12_s03_s06_s10_equal_probability_exploratory_seed42',
 'S15 DenseNet201':ID,
}

OUT = ROOT / 'results/model_comparison/structured/s15_backbone_review'


def main():
    path = ROOT / 'results/structured_experiments' / ID
    config, record = load(path/'config.json'), load(path/'record.json')
    validate_config(config)
    assert record['status'] == 'completed' and config['protocol'] == 'exploratory_image_level'
    assert sha256(ROOT/config['split_manifest']) == config['split_sha256']
    assert config == load(ROOT/'research/configs/phase3'/f'{ID}.json')
    val = pd.read_csv(ROOT/config['split_manifest']).query("split=='val'").set_index('image_id')
    history = pd.read_csv(path/'history.csv')
    assert history.epoch.tolist() == list(range(1, len(history)+1)) and record['epochs'] == len(history)
    checkpoints = ROOT / 'checkpoints/structured' / ID
    latest = torch.load(checkpoints/'latest.pt', map_location='cpu', weights_only=False)
    assert latest['config'] == config
    assert all(torch.isfinite(t).all() for t in latest['model'].values())
    assert np.isfinite(history.select_dtypes(include='number').to_numpy()).all()
    sources = verify_launch_code(latest, load(path/'environment.json'))
    pd.testing.assert_frame_equal(pd.DataFrame(latest['history']), history, check_exact=False, rtol=1e-12, atol=1e-12)
    selected = {}
    for criterion, suffix, name, key in (
        ('accuracy', '', 'best.pt', 'best'),
        ('macro_f1', '_macro_f1', 'best_macro_f1.pt', 'secondary_best'),
    ):
        metrics = load(path/f'validation_metrics{suffix}.json')
        cm = verify_predictions(path/f'validation_predictions{suffix}.csv', metrics, val)
        epoch = int(history.loc[history['val_'+criterion].idxmax(), 'epoch'])
        winner = torch.load(checkpoints/name, map_location='cpu', weights_only=False)
        state = latest[key]
        assert winner['config'] == config and winner['best_epoch'] == state['epoch'] == epoch
        assert winner['selection_metric'] == criterion and winner['metrics'] == state['metrics'] == metrics
        assert all(torch.isfinite(t).all() for t in winner['model'].values())
        assert winner['model'].keys() == state['model'].keys()
        assert all(torch.equal(winner['model'][k], v) for k, v in state['model'].items())
        pd.testing.assert_frame_equal(pd.DataFrame(state['predictions']), pd.read_csv(path/f'validation_predictions{suffix}.csv'), check_exact=False, rtol=1e-12, atol=1e-12)
        row = history.loc[history.epoch == epoch].iloc[0]
        for field in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'loss'):
            assert np.isclose(metrics[field], row['val_'+field], rtol=0, atol=1e-10)
        if not suffix:
            assert record['best_epoch'] == epoch and sha256(checkpoints/name) == record['checkpoint_sha256']
            for field in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1'):
                assert np.isclose(record[field], metrics[field], rtol=0, atol=1e-10)
        folder = path / ('figures_macro_f1' if suffix else 'figures')
        if suffix:
            metric_figures(metrics, folder, 'S15 DenseNet201 | exploratory validation | macro-F1 winner')
        verify_figures(folder, cm, metrics, curves=not suffix)
        selected[criterion] = dict(epoch=epoch, metrics=metrics)

    rows, class_rows, aligned = [], [], {}
    for label, rid in PEERS.items():
        folder = ROOT / 'results/structured_experiments' / rid
        r, c, m = load(folder/'record.json'), load(folder/'config.json'), load(folder/'validation_metrics.json')
        assert r['status'] == 'completed' and r['protocol'] == config['protocol']
        assert r['split_sha256'] == config['split_sha256'] and r['evaluation_split'] == 'validation'
        verify_predictions(folder/'validation_predictions.csv', m, val)
        if r['record_kind'] == 'training_run':
            validate_config(c)
            assert c['selection_metric'] == 'accuracy'
            assert sha256(ROOT/r['checkpoint']) == r['checkpoint_sha256']
        found = pd.read_csv(folder/'validation_predictions.csv').set_index('image_id').loc[val.index]
        aligned[rid] = found.predicted_class.to_numpy()
        rows.append(dict(display_name=label, experiment_id=rid, selection_metric='accuracy' if r['record_kind']=='training_run' else 'fixed parent accuracy winners',
            best_epoch=r['best_epoch'], accuracy=m['accuracy'], macro_precision=m['macro_precision'], macro_recall=m['macro_recall'], macro_f1=m['macro_f1'],
            melanoma_recall=m['per_class']['mel']['recall'], runtime_seconds=r['runtime_seconds'],
            inference_model_passes=2 if r['record_kind']=='fixed_probability_fusion' else 1,
            scope='Same exploratory cohort; seed42; common screening policy for singles; fusion cost differs'))
        for cls in CLASSES:
            class_rows.append(dict(experiment_id=rid, display_name=label, class_name=cls, **m['per_class'][cls]))
    comparison_figures(rows, OUT, 'Same exploratory validation | accuracy-selected singles and fixed S12 fusion\nPretrained model packages differ; no strict/test results')
    write_csv(OUT/'per_class_comparison.csv', class_rows)
    truth = val.diagnosis.to_numpy()
    s06_ok = aligned[ID] == truth
    overlaps = {}
    for label, rid in PEERS.items():
        if rid == ID:
            continue
        other_ok = aligned[rid] == truth
        overlaps[rid] = dict(display_name=label, validation_images=len(val),
            s15_correct=int(s06_ok.sum()), other_correct=int(other_ok.sum()),
            s15_fixes_other=int((s06_ok & ~other_ok).sum()), other_recovers_s15=int((~s06_ok & other_ok).sum()),
            both_wrong=int((~s06_ok & ~other_ok).sum()), both_correct=int((s06_ok & other_ok).sum()),
            prediction_disagreements=int((aligned[ID] != aligned[rid]).sum()),
            oracle_union_accuracy=float((s06_ok | other_ok).mean()),
            scope='Oracle uses true labels: diagnostic upper bound only, not an ensemble or achievable reported result')
    write_json(OUT/'error_overlap.json', overlaps)
    write_csv(OUT/'aligned_errors.csv', [dict(image_id=i, true_class=t, **{rid:str(p[j]) for rid,p in aligned.items()}) for j,(i,t) in enumerate(zip(val.index, truth))])
    baseline = read_registry()
    untouched = [r for r in baseline if r['experiment_id'] != ID]
    record.update(source_availability='original', decision='exclude_from_selected_ensemble_after_marginal_fixed_fusion',
        notes=record['notes'].split(' Closed out on CPU:')[0] + ' Closed out on CPU: both winners, latest history, predictions, class scores, matrices and curves verified. Weaker than ConvNeXt; One equal ConvNeXt/DenseNet fusion reached92.61%/.8771, only2 extra correct vs FP32 reference with lower macro-F1; exclude DenseNet from selected ensemble, no weight search.')
    upsert(record)
    write_json(path/'record.json', record)
    assert untouched == [r for r in read_registry() if r['experiment_id'] != ID]
    report = dict(status='verified_completed', experiment_id=ID, protocol=config['protocol'], epochs=len(history), selected=selected,
        final_epoch_metrics={k:float(history.iloc[-1]['val_'+k]) for k in ('accuracy','macro_precision','macro_recall','macro_f1','loss')},
        final_train_accuracy=float(history.iloc[-1].train_accuracy), minimum_val_loss_epoch=int(history.loc[history.val_loss.idxmin(),'epoch']),
        runtime_seconds=record['runtime_seconds'], peak_allocated_vram_mb=float(history.peak_allocated_vram_mb.max()),
        optimizer_updates=int(history.iloc[-1].optimizer_updates), 
        verified_launch_sources=sources, checkpoint_hashes={p.name:sha256(p) for p in checkpoints.glob('*.pt')},
        source_hashes={n:sha256(path/n) for n in ('config.json','history.csv','validation_metrics.json','validation_metrics_macro_f1.json','validation_predictions.csv','validation_predictions_macro_f1.csv')},
        paired_error_overlap=overlaps, historical_rows_preserved=sum(r['era']=='legacy' for r in untouched),
        all_other_registry_rows_unchanged=True, validation_precision=history.validation_precision.unique().tolist(), gpu_used=False, test_images_loaded=False)
    write_json(path/'closeout_verification.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected','verified_launch_sources','source_hashes','checkpoint_hashes')}, indent=2))


if __name__ == '__main__':
    main()
