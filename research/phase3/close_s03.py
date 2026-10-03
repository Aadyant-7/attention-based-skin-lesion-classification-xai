"""CPU-only S03 closeout and paired S02/S03 exploratory error analysis."""
import json
import numpy as np
import pandas as pd
import torch
from research.common import ROOT, CLASSES, sha256, write_json, write_csv
from research.plots import metric_figures, validate_metrics, comparison_figures
from research.registry import read_registry, upsert
from research.phase3.close_s02 import load, verify_predictions, verify_figures, verify_launch_code

ID = 's03_efficientnet_b0_none_exploratory_seed42'
S02 = 's02_mobilenet_v3_large_none_exploratory_seed42'


def main():
    path = ROOT / 'results/structured_experiments' / ID
    config, record = load(path / 'config.json'), load(path / 'record.json')
    assert record['status'] == 'completed' and config['protocol'] == 'exploratory_image_level'
    assert sha256(ROOT / config['split_manifest']) == config['split_sha256']
    val = pd.read_csv(ROOT / config['split_manifest']).query("split=='val'").set_index('image_id')
    history = pd.read_csv(path / 'history.csv')
    assert history.epoch.tolist() == list(range(1, len(history) + 1))
    assert record['epochs'] == len(history) == 20
    checkpoints = ROOT / 'checkpoints/structured' / ID
    latest = torch.load(checkpoints / 'latest.pt', map_location='cpu', weights_only=False)
    assert latest['config'] == config
    source_verification = verify_launch_code(latest, load(path / 'environment.json'))
    pd.testing.assert_frame_equal(pd.DataFrame(latest['history']), history, check_exact=False, rtol=1e-12, atol=1e-12)
    selected = {}
    for criterion, suffix, name, key in (
        ('accuracy', '', 'best.pt', 'best'),
        ('macro_f1', '_macro_f1', 'best_macro_f1.pt', 'secondary_best'),
    ):
        metrics = load(path / f'validation_metrics{suffix}.json')
        cm = verify_predictions(path / f'validation_predictions{suffix}.csv', metrics, val)
        state = latest[key]
        winner = torch.load(checkpoints / name, map_location='cpu', weights_only=False)
        epoch = int(history.loc[history['val_' + criterion].idxmax(), 'epoch'])
        assert winner['config'] == config and winner['best_epoch'] == state['epoch'] == epoch
        assert winner['selection_metric'] == criterion and winner['metrics'] == state['metrics'] == metrics
        assert all(torch.equal(winner['model'][k], value) for k, value in state['model'].items())
        pd.testing.assert_frame_equal(pd.DataFrame(state['predictions']),
            pd.read_csv(path / f'validation_predictions{suffix}.csv'), check_exact=False, rtol=1e-12, atol=1e-12)
        row = history.loc[history.epoch == epoch].iloc[0]
        for field in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'loss'):
            assert np.isclose(metrics[field], row['val_' + field], rtol=0, atol=1e-10)
        folder = path / ('figures_macro_f1' if suffix else 'figures')
        if suffix:
            metric_figures(metrics, folder, 'S03 EfficientNet-B0 | exploratory validation | macro-F1 winner')
        else:
            assert record['best_epoch'] == epoch and sha256(checkpoints / name) == record['checkpoint_sha256']
            for field in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1'):
                assert np.isclose(record[field], metrics[field], rtol=0, atol=1e-10)
            assert all(torch.equal(latest['model'][k], value) for k, value in state['model'].items())
        verify_figures(folder, cm, metrics, curves=not suffix)
        selected[criterion] = dict(epoch=epoch, metrics=metrics)
    baseline = read_registry()
    untouched = [r for r in baseline if r['experiment_id'] != ID]
    note = ' Closed out on CPU: both selected checkpoints/history/predictions/matrices/class scores verified; matched S02 exploratory comparison; strict results separate.'
    record.update(decision='matched_backbone_control_completed_propose_fixed_fusion', source_availability='original',
                  notes=record['notes'].split(' Closed out on CPU:')[0] + note)
    upsert(record)
    write_json(path / 'record.json', record)
    assert untouched == [r for r in read_registry() if r['experiment_id'] != ID]
    other = ROOT / 'results/structured_experiments' / S02
    c02 = load(other / 'config.json')
    differences = {k: [c02.get(k), config.get(k)] for k in set(c02) | set(config) if c02.get(k) != config.get(k)}
    assert set(differences) == {'experiment_id', 'model', 'weights', 'question'}
    m02 = load(other / 'validation_metrics.json')
    verify_predictions(other / 'validation_predictions.csv', m02, val)
    m03 = selected['accuracy']['metrics']
    out = ROOT / 'results/model_comparison/structured/s02_s03_matched'
    pairs = []
    for c in CLASSES:
        pairs.append(dict(class_name=c, support=m03['per_class'][c]['support'],
            **{f's02_{k}':m02['per_class'][c][k] for k in ('precision', 'recall', 'f1')},
            **{f's03_{k}':m03['per_class'][c][k] for k in ('precision', 'recall', 'f1')},
            f1_delta=m03['per_class'][c]['f1']-m02['per_class'][c]['f1']))
    write_csv(out / 'per_class_comparison.csv', pairs)
    p02 = pd.read_csv(other / 'validation_predictions.csv').set_index('image_id').loc[val.index]
    p03 = pd.read_csv(path / 'validation_predictions.csv').set_index('image_id').loc[val.index]
    ok02, ok03 = p02.predicted_class == val.diagnosis, p03.predicted_class == val.diagnosis
    paired = pd.DataFrame(dict(image_id=val.index, true_class=val.diagnosis.to_numpy(),
        s02_prediction=p02.predicted_class.to_numpy(), s03_prediction=p03.predicted_class.to_numpy(),
        s02_correct=ok02.to_numpy(), s03_correct=ok03.to_numpy()))
    write_csv(out / 'paired_errors.csv', paired.to_dict('records'))
    disagreement = dict(images=len(val), both_correct=int((ok02 & ok03).sum()),
        only_s02_correct=int((ok02 & ~ok03).sum()), only_s03_correct=int((~ok02 & ok03).sum()),
        both_wrong=int((~ok02 & ~ok03).sum()), prediction_disagreements=int((p02.predicted_class != p03.predicted_class).sum()),
        scope='Paired error diagnostics only; no fusion weights or ensemble result selected; not independent lesion-level significance evidence')
    write_json(out / 'paired_error_analysis.json', disagreement)
    rows = []
    for name, rid, m in [('S02 MobileNetV3-Large', S02, m02), ('S03 EfficientNet-B0', ID, m03)]:
        rows.append(dict(display_name=name, experiment_id=rid, accuracy=m['accuracy'], macro_f1=m['macro_f1'],
            melanoma_recall=m['per_class']['mel']['recall'], scope='matched exploratory recipe; accuracy-selected; one seed',
            split_sha256=config['split_sha256']))
    comparison_figures(rows, out, 'Matched exploratory recipe | seed42 | accuracy-selected\nBackbone and pretrained specification differ; no strict results')
    for label, filename, selector in (
        ('Historical weighted B0+CBAM', 'results/exploratory/image_level_weighted_b0_cbam_v1_summary.json', 'best_validation'),
        ('Historical strong-augmentation B0+CBAM', 'results/exploratory/image_level_strong_aug_b0_cbam_v1_summary.json', 'best_validation'),
        ('Historical B0 four-stream ensemble', 'results/exploratory/multires_ensemble/four_equal.json', None),
        ('Historical B0 + PanDerm eight-stream ensemble', 'results/exploratory/panderm_base_image_level_tta_v1/b0_four_plus_four_svc_views_eight_equal.json', None),
    ):
        old = next(r for r in baseline if r['metrics_path'] == filename)
        assert old['split_sha256'] == config['split_sha256'] and old['protocol'] == config['protocol']
        m = load(ROOT / filename); m = m[selector] if selector else m
        assert np.array_equal(validate_metrics(m).sum(1), [m03['per_class'][c]['support'] for c in CLASSES])
        rows.append(dict(display_name=label, experiment_id=old['experiment_id'], accuracy=m['accuracy'],
            macro_f1=m['macro_f1'], melanoma_recall=m['per_class']['mel']['recall'],
            scope='same exploratory cohort; historical recipe/selection/cost differ', split_sha256=old['split_sha256']))
    comparison_figures(rows, ROOT / 'results/model_comparison/structured/s03_exploratory_context',
        'Same exploratory cohort | historical recipe/cost differences\nS02/S03 matched; other rows descriptive, not architecture ablations')
    final_row = history.iloc[-1]
    report = dict(status='verified_completed', experiment_id=ID, epochs=len(history), selected=selected,
        final_epoch=int(final_row.epoch), final_accuracy=float(final_row.val_accuracy), final_macro_f1=float(final_row.val_macro_f1),
        final_loss=float(final_row.val_loss), final_train_accuracy=float(final_row.train_accuracy),
        minimum_val_loss_epoch=int(history.loc[history.val_loss.idxmin(), 'epoch']), runtime_seconds=record['runtime_seconds'],
        paired_errors=disagreement, matched_config_differences=differences, verified_launch_sources=source_verification,
        checkpoint_hashes={p.name:sha256(p) for p in checkpoints.glob('*.pt')},
        source_hashes={name:sha256(path / name) for name in ('config.json', 'history.csv', 'validation_metrics.json',
            'validation_predictions.csv', 'validation_metrics_macro_f1.json', 'validation_predictions_macro_f1.csv')},
        historical_rows_preserved=sum(r['era']=='legacy' for r in untouched),
        all_other_registry_rows_unchanged=True, gpu_used=False, test_images_opened=False)
    write_json(path / 'closeout_verification.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected','source_hashes','verified_launch_sources')}, indent=2))


if __name__ == '__main__':
    main()
