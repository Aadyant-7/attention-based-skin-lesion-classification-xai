"""Fixed PanDerm/B0 probability combinations on strict validation only."""
import csv
import json

import numpy as np
import pandas as pd

from src.data import ROOT, SPLIT
from src.metrics import classification_metrics
from src.utils import write_json


OUTPUT = ROOT / "results/accuracy_exploration/panderm_base_strict_ensembles_v1"


def read(path, ids, labels, key):
    with np.load(path) as saved:
        if not np.array_equal(saved["ids"], ids) or not np.array_equal(saved["y"], labels):
            raise ValueError(f"Cache ordering differs: {path}")
        return saved[key]


def main():
    frame = pd.read_csv(SPLIT)
    val = frame.loc[frame.split == "val"].reset_index(drop=True)
    ids = val.image_id.to_numpy(dtype=str)
    y = val.label.to_numpy()
    cache = ROOT / ".cache"
    features = cache / "panderm_base_strict_lp_v1"
    svm = read(features / "standardized_rbf_svc_c10_val_probabilities.npz", ids, y, "probabilities")
    linear = read(features / "standardized_logreg_c1_val_probabilities.npz", ids, y, "probabilities")
    pilot = read(features / "pilot_best_val_probabilities.npz", ids, y, "probabilities")
    weighted_cache = cache / "phase4/efficientnet_b0_cbam_weighted_v1_val_probabilities.npz"
    oversampled_cache = cache / "phase4/efficientnet_b0_cbam_oversampled_v1_val_probabilities.npz"
    weighted = read(weighted_cache, ids, y, "normal")
    weighted_tta = read(weighted_cache, ids, y, "tta")
    oversampled = read(oversampled_cache, ids, y, "normal")
    candidates = {
        "svm_weighted_equal": (svm + weighted) / 2,
        "svm_weighted_tta_equal": (svm + weighted_tta) / 2,
        "svm_weighted_oversampled_equal": (svm + weighted + oversampled) / 3,
        "linear_weighted_oversampled_equal": (linear + weighted + oversampled) / 3,
        "four_equal": (svm + linear + weighted + oversampled) / 4,
        "pilot_weighted_equal": (pilot + weighted) / 2,
        "pilot_weighted_oversampled_equal": (pilot + weighted + oversampled) / 3,
        "pilot_svm_weighted_oversampled_equal": (pilot + svm + weighted + oversampled) / 4,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, probabilities in candidates.items():
        metrics = classification_metrics(y, probabilities.argmax(axis=1))
        write_json(OUTPUT / f"{name}.json", metrics)
        summary = {"method": name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(OUTPUT / "protocol.json", {
        "strict_validation_images": len(val), "strict_test_evaluated": False,
        "backbone_updated": False,
        "selection_warning": "Eight fixed candidates examined on strict validation; winner is validation-selected",
    })


if __name__ == "__main__":
    main()
