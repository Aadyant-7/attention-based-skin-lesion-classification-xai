"""Decompose validation performance by training-lesion exposure, CPU only.

No model/checkpoint loading, test-label reading, image decoding, or score search.
Uses saved development predictions only. Subgroup scores are diagnostics, not
new independent evaluations. Original scientific files are never overwritten.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from research.common import ROOT, CLASSES, relative, sha256, write_csv, write_json

OUT = ROOT / 'results/diagnostics/generalization_review_2026-10-12'
SOURCES = {
    'S83': dict(split='data/splits/exploratory/image_level_dev_v1.csv',
                predictions='results/short_screening/s83_cbam_f1_addition_cpu/validation_predictions.csv',
                metrics='results/short_screening/s83_cbam_f1_addition_cpu/validation_metrics.json',
                selector='original frozen equal-six; validation-selected constituent checkpoints'),
    'R202_EMA20': dict(split='data/splits/image_level_replication/v2/fold_00.csv',
                      predictions='results/image_level_replication/v2/r202_convnextv2_base_fold00_seed42/best_accuracy/predictions.csv',
                      metrics='results/image_level_replication/v2/r202_convnextv2_base_fold00_seed42/best_accuracy/validation_metrics.json',
                      selector='original raw-or-EMA accuracy selector: EMA epoch20'),
}


def development_only(path):
    # Read test image/lesion IDs for exclusion only; exclude its diagnosis/label rows.
    ids = pd.read_csv(path, usecols=['image_id', 'lesion_id', 'split'])
    excluded = set((ids.index[ids.split == 'test'] + 1).tolist())
    dev = pd.read_csv(path, skiprows=lambda row: row in excluded)
    assert set(dev.split) == {'train', 'val'} and dev.image_id.is_unique
    assert not set(dev.image_id) & set(ids.loc[ids.split == 'test', 'image_id'])
    return dev


def score(frame):
    y, pred = frame.true_class.to_numpy(), frame.predicted_class.to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    return dict(images=len(frame), lesions=int(frame.lesion_id.nunique()),
                correct=int((y == pred).sum()), accuracy=float((y == pred).mean()),
                macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                macro_f1=float(f1.mean()), balanced_accuracy=float(recall.mean()),
                confusion_matrix=confusion_matrix(y, pred, labels=CLASSES).tolist(),
                per_class={c: dict(precision=float(precision[i]), recall=float(recall[i]),
                                   f1=float(f1[i]), support=int(support[i])) for i, c in enumerate(CLASSES)})


def run():
    completion = OUT / 'audit.json'
    if completion.exists():
        audit = json.loads(completion.read_text())
        assert all(sha256(ROOT / p) == digest for p, digest in audit['source_sha256'].items())
        print('Completed diagnosis preserved; source hashes still match')
        return
    sources, studies, summary, class_rows, confidence_rows = {}, {}, [], [], []
    for name, config in SOURCES.items():
        for key in ['split', 'predictions', 'metrics']:
            sources[config[key]] = sha256(ROOT / config[key])
        dev = development_only(ROOT / config['split'])
        train, val = dev.loc[dev.split == 'train'], dev.loc[dev.split == 'val']
        frame = pd.read_csv(ROOT / config['predictions'])
        assert frame.image_id.tolist() == val.image_id.tolist()
        assert frame.lesion_id.tolist() == val.lesion_id.tolist()
        assert frame.true_class.tolist() == val.diagnosis.tolist()
        p = frame[['p_' + c for c in CLASSES]].to_numpy()
        assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
        assert [CLASSES[i] for i in p.argmax(1)] == frame.predicted_class.tolist()
        seen = frame.lesion_id.isin(set(train.lesion_id))
        groups = {'all_validation': frame, 'training_lesion_represented': frame.loc[seen],
                  'training_lesion_absent': frame.loc[~seen]}
        results = {group: score(a) for group, a in groups.items()}
        saved = json.loads((ROOT / config['metrics']).read_text())
        for key in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']:
            assert np.isclose(results['all_validation'][key], saved[key], atol=1e-12)
        total_support = results['all_validation']['per_class']
        for group, result in results.items():
            assert all(result['per_class'][c]['support'] > 0 for c in CLASSES)
            standardized = sum(total_support[c]['support'] * result['per_class'][c]['recall'] for c in CLASSES) / len(frame)
            row = dict(method=name, subgroup=group, **{k: result[k] for k in
                ['images', 'lesions', 'correct', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'balanced_accuracy']},
                melanoma_recall=result['per_class']['mel']['recall'], akiec_recall=result['per_class']['akiec']['recall'],
                nv_fraction=result['per_class']['nv']['support'] / result['images'],
                accuracy_standardized_to_full_validation_class_mix=standardized,
                evaluation_role='posthoc_validation_diagnostic_not_independent_test')
            summary.append(row)
            class_rows.extend(dict(method=name, subgroup=group, class_name=c, **result['per_class'][c]) for c in CLASSES)
        # These IDs and probabilities were already known; no labels from test are accessed.
        confidence = p.max(1)
        correct = frame.true_class == frame.predicted_class
        for mask_name, mask in [('correct', correct), ('incorrect', ~correct)]:
            confidence_rows.append(dict(method=name, subset=mask_name, images=int(mask.sum()),
                                         mean_top_probability=float(confidence[mask].mean()),
                                         errors_or_correct_above_0_9=int((mask & (confidence >= .9)).sum())))
        studies[name] = dict(selector=config['selector'], training_images=len(train), validation_images=len(val),
                             groups=results, prediction_source=config['predictions'])
    history_path = ROOT / 'results/image_level_replication/v2/r202_convnextv2_base_fold00_seed42/history.csv'
    sources[relative(history_path)] = sha256(history_path)
    history = pd.read_csv(history_path).set_index('epoch')
    late = {}
    for epoch in [20, 40, 50]:
        late[str(epoch)] = {key: float(history.loc[epoch, key]) for key in
            ['train_accuracy', 'train_loss', 'val_accuracy', 'val_macro_f1', 'val_loss',
             'ema_val_accuracy', 'ema_val_macro_f1', 'ema_val_loss']}
    write_csv(OUT / 'validation_subgroups.csv', summary)
    write_csv(OUT / 'validation_subgroup_classes.csv', class_rows)
    write_csv(OUT / 'validation_confidence.csv', confidence_rows)
    audit = dict(status='completed', purpose='diagnose apparent generalization gap from already-saved development evidence',
                 studies=studies, r202_learning_curve_snapshots=late, source_sha256=sources,
                 training=False, gpu_used=False, test_images_loaded=False, test_labels_loaded=False,
                 checkpoint_loaded=False, predictions_changed=False, original_metrics_changed=False,
                 limitations=['Both validation cohorts were already used for checkpoint/recipe selection.',
                              'Seen/unseen lesion subgroups have very different class mixtures and small minority supports.',
                              'Association is not a controlled causal effect of overlap; neither subgroup is a fresh test.',
                              'Class standardization adjusts prevalence only, not image difficulty or prior selection bias.'])
    write_json(completion, audit)
    print(pd.DataFrame(summary)[['method', 'subgroup', 'images', 'accuracy', 'macro_f1', 'melanoma_recall', 'nv_fraction']].to_string(index=False))
    print('Saved validation-only diagnosis; no training, image inference, or test labels loaded.')


if __name__ == '__main__':
    run()
