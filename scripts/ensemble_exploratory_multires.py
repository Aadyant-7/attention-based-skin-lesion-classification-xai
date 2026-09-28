"""Small prespecified validation-only ensemble of frozen exploratory checkpoints."""
import csv
import json

import numpy as np
import pandas as pd

from src.data import ROOT
from src.metrics import classification_metrics
from src.utils import write_json


CACHE = ROOT / ".cache/exploratory_multires"
OUTPUT = ROOT / "results/exploratory/multires_ensemble"
BASE = "image_level_weighted_b0_cbam_v1"
STRONG = "image_level_strong_aug_b0_cbam_v1"


def read(run, size):
    with np.load(CACHE / f"{run}_{size}.npz") as data:
        return data["ids"], data["y"], data["probabilities"]


def main():
    streams = {}
    reference_ids = reference_y = None
    for key, run, size in (("base224", BASE, 224), ("base320", BASE, 320), ("base384", BASE, 384),
                           ("strong224", STRONG, 224), ("strong320", STRONG, 320), ("strong384", STRONG, 384)):
        ids, y, probs = read(run, size)
        if reference_ids is None:
            reference_ids, reference_y = ids, y
        elif not np.array_equal(ids, reference_ids) or not np.array_equal(y, reference_y):
            raise ValueError("Cached stream ordering/labels differ")
        streams[key] = probs
    strict = pd.read_csv(ROOT / "data/splits/split_assignments.csv")
    assert not set(reference_ids) & set(strict.loc[strict.split == "test", "image_id"])
    candidates = {
        "base224": {"base224": 1.0},
        "base224_base384": {"base224": .5, "base384": .5},
        "base224_strong224": {"base224": .5, "strong224": .5},
        "three_equal": {"base224": 1/3, "base384": 1/3, "strong224": 1/3},
        "four_equal": {"base224": .25, "base384": .25, "strong224": .25, "strong384": .25},
        "six_equal": {key: 1/6 for key in streams},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, weights in candidates.items():
        probabilities = sum(weight * streams[key] for key, weight in weights.items())
        metrics = classification_metrics(reference_y, probabilities.argmax(axis=1))
        summary = {"method": name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                   "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"]}
        summaries.append(summary)
        write_json(OUTPUT / f"{name}.json", {"weights": weights, **metrics})
        print(json.dumps(summary), flush=True)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(OUTPUT / "protocol.json", {"strict_test_evaluated": False,
        "model_weights_updated": False, "validation_images": len(reference_ids),
        "selection_warning": "All candidates examined on the same exploratory image-level validation split; selected maximum may be optimistic"})
    # Compare the published singleton-lesion test selection with overlapping lesions.
    frame = pd.read_csv(ROOT / "data/splits/exploratory/image_level_dev_v1.csv")
    val = frame.loc[frame.split == "val"].set_index("image_id").loc[reference_ids]
    train_lesions = set(frame.loc[frame.split == "train", "lesion_id"])
    lesion_sizes = frame.groupby("lesion_id").size()
    singleton = val.lesion_id.map(lesion_sizes).to_numpy() == 1
    shared = val.lesion_id.isin(train_lesions).to_numpy()
    four = sum(weight * streams[key] for key, weight in candidates["four_equal"].items())
    groups = {"singleton_lesions": singleton, "shared_train_lesion": shared,
              "multi_image_unseen_lesion": ~(singleton | shared)}
    subgroup = {}
    for name, mask in groups.items():
        y = reference_y[mask]
        predicted = four[mask].argmax(axis=1)
        subgroup[name] = {"images": int(mask.sum()), **classification_metrics(y, predicted)}
        print(json.dumps({"group": name, "images": int(mask.sum()),
                          "accuracy": subgroup[name]["accuracy"],
                          "macro_f1": subgroup[name]["macro_f1"]}), flush=True)
    write_json(OUTPUT / "subgroup_audit.json", subgroup)


if __name__ == "__main__":
    main()
