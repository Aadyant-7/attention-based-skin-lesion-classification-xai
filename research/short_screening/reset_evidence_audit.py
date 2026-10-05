"""CPU-only development input and saved-prediction audit; no model inference."""
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from research.common import ROOT, CLASSES
from research.strict_protocol import development_data, partition_metadata


def inspect_image(row):
    try:
        path = Path(row.path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            size, mode = image.size, image.mode
            image.verify()
        return dict(image_id=row.image_id, split=row.split, diagnosis=row.diagnosis,
                    sha256=digest, size=str(size), mode=mode, error=None)
    except Exception as exc:
        return dict(image_id=row.image_id, error=str(exc))


def main():
    config = json.loads((ROOT/'research/short_screening/targeted_finetune_v1.json').read_text())
    train, val, _ = development_data(config)
    _, split_report = partition_metadata()
    with ThreadPoolExecutor(max_workers=4) as pool:
        images = list(pool.map(inspect_image, pd.concat([train, val]).itertuples(index=False)))
    valid = [r for r in images if r['error'] is None]
    hashes = {}
    for row in valid:
        hashes.setdefault(row['sha256'], []).append(row)
    duplicated = [group for group in hashes.values() if len(group) > 1]
    checks = []
    for rid in ['s28_efficientnet_b0_final_strict_seed42',
                's29_convnext_tiny_final_strict_seed42',
                's30_efficientnet_v2_s_final_strict_seed42']:
        folder = ROOT/'results/structured_experiments'/rid
        frame = pd.read_csv(folder/'validation_predictions.csv')
        assert not frame.image_id.duplicated().any()
        assert set(frame.image_id) == set(val.image_id)
        frame = frame.set_index('image_id').loc[val.image_id]
        assert frame.true_class.tolist() == val.diagnosis.tolist()
        probabilities = frame[[f'p_{cl}' for cl in CLASSES]].to_numpy()
        assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1, atol=1e-5)
        predictions = np.asarray(CLASSES)[probabilities.argmax(1)]
        assert (predictions == frame.predicted_class).all()
        metrics = json.loads((folder/'validation_metrics.json').read_text())
        accuracy = accuracy_score(frame.true_class, predictions)
        macro_f1 = f1_score(frame.true_class, predictions, labels=CLASSES, average='macro')
        assert abs(accuracy-metrics['accuracy']) < 1e-10
        assert abs(macro_f1-metrics['macro_f1']) < 1e-10
        checks.append(dict(run=rid, accuracy=accuracy, macro_f1=macro_f1,
                           labels_probabilities_and_metrics_consistent=True))
    report = dict(split=split_report, development_images_checked=len(images),
                  corrupt_or_unreadable=[r for r in images if r['error']],
                  dimensions=dict(Counter(r['size'] for r in valid)),
                  modes=dict(Counter(r['mode'] for r in valid)),
                  exact_duplicate_groups=duplicated,
                  cross_train_validation_exact_duplicate_groups=[g for g in duplicated if len({r['split'] for r in g})>1],
                  saved_prediction_checks=checks, gpu_used=False, test_inference=False,
                  scope='Header/integrity, byte duplicates, identities and saved metrics; not a visual label-quality audit or perceptual duplicate audit.')
    out = ROOT/'results/short_screening/reset_evidence_audit'
    out.mkdir(parents=True, exist_ok=True)
    (out/'input_and_prediction_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['exact_duplicate_groups','split']}, indent=2))


if __name__ == '__main__':
    main()
