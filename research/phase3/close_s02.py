"""Verify completed S02 using saved artifacts only; no images/models/GPU."""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from research.common import ROOT, CLASSES, sha256, write_json
from research.plots import validate_metrics, metric_figures, comparison_figures
from research.registry import read_registry, upsert

ID = 's02_mobilenet_v3_large_none_exploratory_seed42'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_predictions(path, metrics, validation):
    predictions = pd.read_csv(path)
    assert predictions.image_id.is_unique and len(predictions) == len(validation)
    assert set(predictions.image_id) == set(validation.index)
    found = predictions.set_index('image_id').loc[validation.index]
    assert found.true_class.equals(validation.diagnosis)
    probabilities = found[[f'p_{c}' for c in CLASSES]].to_numpy()
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(1), 1, atol=1e-6)
    predicted = np.asarray(CLASSES)[probabilities.argmax(1)]
    assert np.array_equal(predicted, found.predicted_class)
    cm = validate_metrics(metrics)
    assert np.array_equal(cm, confusion_matrix(validation.diagnosis, predicted, labels=list(CLASSES)))
    return cm


def verify_figures(folder, cm, metrics, curves=False):
    names = ['confusion_matrix', 'confusion_matrix_normalized', 'per_class_metrics', 'class_support']
    if curves:
        names += ['accuracy_curves', 'loss_curves', 'macro_f1_curve']
    for name in names:
        assert (folder / (name + '.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert (folder / (name + '.pdf')).read_bytes().startswith(b'%PDF')
    stored = pd.read_csv(folder / 'confusion_matrix.csv').set_index('true_class').loc[list(CLASSES), list(CLASSES)].to_numpy()
    assert np.array_equal(stored, cm)
    normalized = pd.read_csv(folder / 'confusion_matrix_normalized.csv').set_index('true_class').loc[list(CLASSES), list(CLASSES)].to_numpy()
    assert np.allclose(normalized, cm / cm.sum(1, keepdims=True))
    classes = pd.read_csv(folder / 'per_class_metrics.csv').set_index('class')
    for c in CLASSES:
        for key, value in metrics['per_class'][c].items():
            assert np.isclose(classes.loc[c, key], value)


def main():
    path = ROOT / 'results/structured_experiments' / ID
    config = load(path / 'config.json')
    record = load(path / 'record.json')
    assert record['status'] == 'completed'
    assert config['protocol'] == 'exploratory_image_level'
    manifest = ROOT / config['split_manifest']
    assert sha256(manifest) == config['split_sha256']
    val = pd.read_csv(manifest).query("split=='val'").set_index('image_id')
    history = pd.read_csv(path / 'history.csv')
    assert history.epoch.tolist() == list(range(1, len(history) + 1))
    assert record['epochs'] == len(history)
    checkpoint_dir = ROOT / 'checkpoints/structured' / ID
    latest = torch.load(checkpoint_dir / 'latest.pt', map_location='cpu', weights_only=False)
    assert latest['config'] == config
    for filename, digest in latest['code_hashes'].items():
        assert sha256(ROOT / filename) == digest
    pd.testing.assert_frame_equal(pd.DataFrame(latest['history']), history, check_exact=False, rtol=1e-12, atol=1e-12)
    results = {}
    for criterion, suffix, checkpoint_name, state_key in (
        ('accuracy', '', 'best.pt', 'best'),
        ('macro_f1', '_macro_f1', 'best_macro_f1.pt', 'secondary_best'),
    ):
        metrics = load(path / f'validation_metrics{suffix}.json')
        cm = verify_predictions(path / f'validation_predictions{suffix}.csv', metrics, val)
        epoch = int(history.loc[history['val_' + criterion].idxmax(), 'epoch'])
        state = latest[state_key]
        selected = torch.load(checkpoint_dir / checkpoint_name, map_location='cpu', weights_only=False)
        assert selected['config'] == config and selected['best_epoch'] == state['epoch'] == epoch
        assert selected['metrics'] == state['metrics'] == metrics
        assert selected['selection_metric'] == criterion
        assert all(torch.equal(selected['model'][k], value) for k, value in state['model'].items())
        best_row = history.loc[history.epoch == epoch].iloc[0]
        for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'loss'):
            assert np.isclose(metrics[key], best_row['val_' + key], rtol=0, atol=1e-10)
        if suffix:
            figures = path / 'figures_macro_f1'
            metric_figures(metrics, figures, 'S02 MobileNetV3-Large | exploratory validation | macro-F1 winner')
        else:
            figures = path / 'figures'
            assert record['best_epoch'] == epoch
            assert sha256(checkpoint_dir / checkpoint_name) == record['checkpoint_sha256']
            for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1'):
                assert np.isclose(record[key], metrics[key])
        verify_figures(figures, cm, metrics, curves=not suffix)
        results[criterion] = dict(epoch=epoch, metrics=metrics, checkpoint_sha256=sha256(checkpoint_dir / checkpoint_name))
    legacy_before = [r for r in read_registry() if r['era'] == 'legacy']
    note = ' Closed out on CPU: both selected checkpoints/predictions/history/matrices/class scores verified. Comparisons restricted to the same exploratory manifest; legacy recipes are descriptive context.'
    record.update(decision='exploratory_screen_completed_pending_next_approval', source_availability='original',
                  notes=record['notes'].split(' Closed out on CPU:')[0] + note)
    upsert(record)
    write_json(path / 'record.json', record)
    assert legacy_before == [r for r in read_registry() if r['era'] == 'legacy']
    sources = [
        ('Historical B0+CBAM | weighted single model', 'results/exploratory/image_level_weighted_b0_cbam_v1_summary.json', 'best_validation'),
        ('Historical B0+CBAM | strong augmentation', 'results/exploratory/image_level_strong_aug_b0_cbam_v1_summary.json', 'best_validation'),
        ('Historical B0+CBAM | four-stream ensemble', 'results/exploratory/multires_ensemble/four_equal.json', None),
        ('Historical B0+CBAM + PanDerm | eight streams', 'results/exploratory/panderm_base_image_level_tta_v1/b0_four_plus_four_svc_views_eight_equal.json', None),
    ]
    primary = results['accuracy']['metrics']
    rows = [dict(display_name='S02 MobileNetV3-Large | single model', experiment_id=ID, accuracy=primary['accuracy'],
                 macro_f1=primary['macro_f1'], melanoma_recall=primary['per_class']['mel']['recall'],
                 scope='structured; accuracy-selected', source_path=(path / 'validation_metrics.json').relative_to(ROOT).as_posix(),
                 split_sha256=config['split_sha256'])]
    for name, filename, selector in sources:
        old_record = next(r for r in legacy_before if r['metrics_path'] == filename)
        assert old_record['split_sha256'] == config['split_sha256']
        old = load(ROOT / filename)
        metrics = old[selector] if selector else old
        cm = validate_metrics(metrics)
        assert np.array_equal(cm.sum(1), [primary['per_class'][c]['support'] for c in CLASSES])
        rows.append(dict(display_name=name, experiment_id=old_record['experiment_id'], accuracy=metrics['accuracy'],
                         macro_f1=metrics['macro_f1'], melanoma_recall=metrics['per_class']['mel']['recall'],
                         scope='same exploratory cohort; different recipe/selection/cost; historical context',
                         source_path=filename, split_sha256=old_record['split_sha256']))
    comparison_figures(rows, ROOT / 'results/model_comparison/structured/s02_exploratory_context',
                       'Same exploratory validation cohort | different recipes/costs\nHistorical context, not controlled architecture ranking')
    report = dict(status='verified_completed', experiment_id=ID, protocol=config['protocol'], epochs=len(history),
                  selected=results, runtime_seconds=record['runtime_seconds'],
                  checkpoint_hashes={p.name: sha256(p) for p in checkpoint_dir.glob('*.pt')},
                  source_hashes={name: sha256(path / name) for name in ('config.json', 'history.csv', 'validation_metrics.json',
                      'validation_predictions.csv', 'validation_metrics_macro_f1.json', 'validation_predictions_macro_f1.csv')},
                  historical_registry_rows_preserved=len(legacy_before),
                  failed_launch_log='results/audit/s02_pretraining_download_failed_20261003/train.log',
                  verified_checks=['both selected checkpoints', 'history', 'earliest criterion maxima',
                      'both 1503-image prediction tables', 'matrices/class scores/figures', 'same exploratory comparison split'],
                  gpu_used=False, test_images_opened=False)
    write_json(path / 'closeout_verification.json', report)
    print(json.dumps(dict(status=report['status'], epochs=len(history), selected=results, runtime_seconds=record['runtime_seconds']), indent=2))


if __name__ == '__main__':
    main()
