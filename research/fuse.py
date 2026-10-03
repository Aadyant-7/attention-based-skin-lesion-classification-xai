"""Predeclared equal-probability fusions of saved exploratory predictions; CPU only."""
import argparse
import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, CLASSES, sha256, write_json, write_csv
from .plots import validate_metrics, metric_figures, comparison_figures
from .registry import FIELDS, read_registry, upsert

S02 = 's02_mobilenet_v3_large_none_exploratory_seed42'
S03 = 's03_efficientnet_b0_none_exploratory_seed42'
S06 = 's06_convnext_tiny_none_exploratory_seed42'
CANDIDATES = {
    's17_s06_s15_equal_probability_exploratory_seed42': [S06, 's15_densenet201_none_exploratory_seed42'],
    's11_s06_s10_equal_probability_exploratory_seed42': ['s06_convnext_tiny_none_exploratory_seed42', 's10_efficientnet_v2_s_none_exploratory_seed42'],
    's12_s03_s06_s10_equal_probability_exploratory_seed42': ['s03_efficientnet_b0_none_exploratory_seed42', 's06_convnext_tiny_none_exploratory_seed42', 's10_efficientnet_v2_s_none_exploratory_seed42'],
    's04_s02_s03_equal_probability_exploratory_seed42': [S02, S03],
    's07_s02_s06_equal_probability_exploratory_seed42': [S02, S06],
    's08_s03_s06_equal_probability_exploratory_seed42': [S03, S06],
    's09_s02_s03_s06_equal_probability_exploratory_seed42': [S02, S03, S06],
}


def validate_candidate(config):
    expected = CANDIDATES.get(config['experiment_id'])
    if (not expected or config['recipe_version'] != 'equal_probability_v1'
            or config['parent_run_ids'] != expected
            or config['weights'] != [1/len(expected)]*len(expected)
            or config['protocol'] != 'exploratory_image_level'
            or config['class_order'] != list(CLASSES)):
        raise ValueError('Only the predeclared equal-fusion candidates are supported; no weight/subset search')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def aligned_probabilities(frame, validation):
    if not frame.image_id.is_unique or set(frame.image_id) != set(validation.index):
        raise ValueError('Prediction identities differ from validation manifest')
    aligned = frame.set_index('image_id').loc[validation.index]
    if not aligned.true_class.equals(validation.diagnosis):
        raise ValueError('Prediction labels differ from manifest')
    probabilities = aligned[[f'p_{c}' for c in CLASSES]].to_numpy()
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-6)):
        raise ValueError('Invalid probabilities')
    if not np.array_equal(np.asarray(CLASSES)[probabilities.argmax(1)], aligned.predicted_class):
        raise ValueError('Predictions disagree with probabilities')
    return probabilities


def preflight(config):
    validate_candidate(config)
    manifest = ROOT / config['split_manifest']
    if sha256(manifest) != config['split_sha256']:
        raise ValueError('Exploratory manifest changed')
    strict = ROOT / 'data/splits/split_assignments.csv'
    if sha256(strict) != config['locked_manifest_sha256']:
        raise ValueError('Original locked manifest changed')
    frame, original = pd.read_csv(manifest), pd.read_csv(strict)
    if frame.image_id.duplicated().any():
        raise ValueError('Duplicate manifest images')
    test = frame.query("split=='test'").set_index('image_id').sort_index()
    original_test = original.query("split=='test'").set_index('image_id').sort_index()
    pd.testing.assert_frame_equal(test[['label','diagnosis','lesion_id']], original_test[['label','diagnosis','lesion_id']])
    validation = frame.query("split=='val'").set_index('image_id')
    parents, probabilities = [], []
    for rid in config['parent_run_ids']:
        folder = ROOT / 'results/structured_experiments' / rid
        parent, recipe = read(folder / 'record.json'), read(folder / 'config.json')
        if (parent['status'] != 'completed' or parent['protocol'] != config['protocol']
                or parent['split_sha256'] != config['split_sha256'] or recipe['selection_metric'] != 'accuracy'
                or recipe['split_sha256'] != config['split_sha256']
                or parent['checkpoint_sha256'] != config['checkpoint_sha256'][rid]
                or sha256(folder / 'validation_predictions.csv') != config['prediction_sha256'][rid]):
            raise ValueError('Parent source/selection/protocol changed')
        m = read(folder / 'validation_metrics.json'); validate_metrics(m)
        if sha256(folder/'validation_metrics.json') != parent['source_sha256']:
            raise ValueError('Parent metrics provenance changed')
        probabilities.append(aligned_probabilities(pd.read_csv(folder / 'validation_predictions.csv'), validation))
        parents.append((parent, m))
    return frame, validation, parents, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--repair', action='store_true', help='Recompute the same CPU candidate after a partial artifact write')
    args = parser.parse_args()
    config = read(args.config)
    frame, validation, parents, probabilities = preflight(config)
    out = ROOT / 'results/structured_experiments' / config['experiment_id']
    if args.check:
        print(json.dumps(dict(status='prepared_cpu_candidate', validation_images=len(validation),
            output=out.relative_to(ROOT).as_posix(), gpu_used=False, test_images_loaded=False,
            raw_images_required=False, checkpoints_required=False), indent=2)); return
    if out.exists():
        if read(out / 'config.json') != config:
            raise ValueError('Existing ID belongs to another config')
        if (out / 'record.json').exists() and read(out / 'record.json')['status'] == 'completed':
            print('Already completed; preserved without recomputation.'); return
        if not args.repair:
            raise ValueError('Partial output exists; inspect then use --repair for the same CPU candidate')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', config)
    row = {field:'' for field in FIELDS}
    row.update(experiment_id=config['experiment_id'], era='structured', record_kind='fixed_probability_fusion',
        phase='ensemble_screening', protocol=config['protocol'], evaluation_split='validation',
        split_manifest=config['split_manifest'], split_sha256=config['split_sha256'], method=config['question'],
        model='+'.join(p['model'] for p,m in parents), attention='none', image_size=224, seed=42,
        ensemble_members=json.dumps(config['parent_run_ids']), ensemble_weights=json.dumps(config['weights']),
        checkpoint=json.dumps([p['checkpoint'] for p,m in parents]), checkpoint_available_local=all((ROOT/p['checkpoint']).exists() for p,m in parents),
        config_path=(out/'config.json').relative_to(ROOT).as_posix(), status='running')
    upsert(row); write_json(out / 'record.json', row)
    logger = logging.getLogger(config['experiment_id']); logger.setLevel(logging.INFO); logger.propagate=False
    handlers = [logging.FileHandler(out / 'run.log', encoding='utf-8'), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s')); logger.addHandler(handler)
    start = time.perf_counter()
    try:
        logger.info('START CPU-only fixed equal fusion of %d parents; no images/models/test loader; no weight search', len(parents))
        fused = sum(w*p for w,p in zip(config['weights'], probabilities))
        labels = validation.diagnosis.map(dict(zip(CLASSES, range(7)))).to_numpy()
        counts = frame.query("split=='train'").diagnosis.value_counts()
        class_weights = np.sqrt(len(frame.query("split=='train'")) / np.asarray([counts[c] for c in CLASSES]))
        class_weights /= class_weights.mean()
        loss = float(np.average(-np.log(np.clip(fused[np.arange(len(labels)), labels], 1e-12, 1)), weights=class_weights[labels]))
        from .train import metric_report  # pure CPU array metrics; never calls the training runner
        metrics = metric_report(labels, fused, loss); validate_metrics(metrics)
        write_json(out / 'validation_metrics.json', metrics)
        predictions = [dict(image_id=i, true_class=CLASSES[y], predicted_class=CLASSES[int(p.argmax())],
            **{f'p_{c}':float(p[j]) for j,c in enumerate(CLASSES)}) for i,y,p in zip(validation.index, labels, fused)]
        write_csv(out / 'validation_predictions.csv', predictions)
        aligned_probabilities(pd.read_csv(out / 'validation_predictions.csv'), validation)
        label = config['experiment_id'].split('_',1)[0].upper() + ' fixed equal fusion'
        metric_figures(metrics, out / 'figures', label + ' | exploratory validation | no weight tuning')
        write_json(out / 'parent_artifacts.json', dict(parent_runs=config['parent_run_ids'],
            training_curves=[str(Path(p['history_path']).parent/'figures') for p,m in parents],
            parent_checkpoints=[p['checkpoint'] for p,m in parents],
            note='No training, new checkpoint, training curve or best epoch exists for this CPU fusion. Parent models required for future image inference.'))
        compare = [dict(display_name=p['model'], accuracy=m['accuracy'], macro_f1=m['macro_f1']) for p,m in parents]
        compare.append(dict(display_name=label, accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1']))
        comparison_figures(compare, ROOT/'results/model_comparison/structured'/f"{config['experiment_id'].split('_',1)[0]}_fixed_fusion", 'Same exploratory validation | single models and fixed equal fusion')
        row.update(status='completed', runtime_seconds=time.perf_counter()-start,
            **{k:metrics[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')}, val_loss=loss,
            metrics_path=(out/'validation_metrics.json').relative_to(ROOT).as_posix(),
            source_sha256=sha256(out/'validation_metrics.json'), source_availability='original',
            plots_dir=(out/'figures').relative_to(ROOT).as_posix(),
            confusion_matrix_path=(out/'figures/confusion_matrix.csv').relative_to(ROOT).as_posix(),
            decision='fixed_fusion_completed_pending_review',
            notes='One predeclared equal-probability candidate, parent accuracy winners; validation-selected parent models; exploratory only; no GPU/raw/test images/new checkpoint.')
        upsert(row); write_json(out / 'record.json', row)
        logger.info('COMPLETED accuracy=%.6f macro_f1=%.6f', metrics['accuracy'], metrics['macro_f1'])
    except BaseException as exc:
        row.update(status='failed', notes=repr(exc)); upsert(row); write_json(out/'record.json', row)
        logger.exception('FAILED; parent artifacts preserved'); raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler); handler.close()


if __name__ == '__main__':
    main()
