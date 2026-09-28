"""Check fixed PanDerm orientation ensembles on strict validation only."""
import csv
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from src.data import ROOT, SPLIT, image_paths
from src.metrics import classification_metrics
from src.utils import write_json
from scripts.probe_panderm_base import TRANSFORM, load_backbone
from scripts.probe_panderm_image_level_tta import flipped_features


CACHE = ROOT / ".cache/panderm_base_strict_lp_v1"
OUTPUT = ROOT / "results/accuracy_exploration/panderm_base_strict_tta_v1"
SVM = ROOT / "checkpoints/accuracy_exploration/panderm_base_strict_heads_v1/standardized_rbf_svc_c10.joblib"


def read(path, ids, y, key):
    with np.load(path) as saved:
        if not np.array_equal(saved["ids"], ids) or not np.array_equal(saved["y"], y):
            raise ValueError(f"Cache order differs: {path}")
        return saved[key]


def main():
    frame = pd.read_csv(SPLIT)
    train = frame.loc[frame.split == "train"]
    val = frame.loc[frame.split == "val"].reset_index(drop=True)
    test = frame.loc[frame.split == "test"]
    if len(val) != 1503 or set(train.lesion_id) & set(val.lesion_id) or set(val.lesion_id) & set(test.lesion_id):
        raise ValueError("Strict lesion-disjoint validation changed")
    paths, duplicates = image_paths()
    if duplicates or any(image_id not in paths for image_id in val.image_id):
        raise ValueError("Missing or duplicate validation images")
    ids, y = val.image_id.to_numpy(dtype=str), val.label.to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_backbone(device)
    normal = read(CACHE / "val.npz", ids, y, "features")
    transforms_by_name = {
        "hflip": transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), TRANSFORM]),
        "vflip": transforms.Compose([transforms.RandomVerticalFlip(p=1.0), TRANSFORM]),
        "hvflip": transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                                       transforms.RandomVerticalFlip(p=1.0), TRANSFORM]),
    }
    features = {"normal": normal}
    for name, transform in transforms_by_name.items():
        features[name] = flipped_features(model, val, paths, name, transform, device, CACHE)
    del model
    head = joblib.load(SVM)
    if not np.array_equal(head[-1].classes_, np.arange(7)):
        raise ValueError("PanDerm SVM class order changed")
    svc_views = {name: head.predict_proba(x) for name, x in features.items()}
    svc_normal = svc_views["normal"]
    svc_tta = sum(svc_views.values()) / 4
    w = read(ROOT / ".cache/phase4/efficientnet_b0_cbam_weighted_v1_val_probabilities.npz", ids, y, "normal")
    o = read(ROOT / ".cache/phase4/efficientnet_b0_cbam_oversampled_v1_val_probabilities.npz", ids, y, "normal")
    pilot = read(CACHE / "pilot_best_val_probabilities.npz", ids, y, "probabilities")
    candidates = {
        "prior_four_control": (w + o + pilot + svc_normal) / 4,
        "svc_tta4": svc_tta,
        "weighted_b0_plus_svc_tta_two_equal": (w + svc_tta) / 2,
        "two_b0_plus_svc_tta_three_equal": (w + o + svc_tta) / 3,
        "prior_four_with_svc_tta_equal": (w + o + pilot + svc_tta) / 4,
        "two_b0_plus_four_svc_views_six_equal": (w + o + sum(svc_views.values())) / 6,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, probabilities in candidates.items():
        metrics = classification_metrics(y, probabilities.argmax(axis=1))
        write_json(OUTPUT / f"{name}.json", metrics)
        row = {"method": name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
        summaries.append(row)
        print(json.dumps(row), flush=True)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(OUTPUT / "protocol.json", {"strict_validation_images": len(val),
        "strict_test_evaluated": False, "model_weights_updated": False,
        "validation_warning": "Six fixed candidates on strict validation; winner validation-selected"})


if __name__ == "__main__":
    main()
