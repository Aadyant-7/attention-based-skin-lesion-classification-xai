"""Frozen PanDerm heads and fixed B0 ensembles on exploratory image-level validation."""
import csv
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import ROOT, SPLIT, image_paths
from src.metrics import classification_metrics
from src.utils import write_json
from scripts.probe_panderm_base import features, load_backbone


SPLIT_PATH = ROOT / "data/splits/exploratory/image_level_dev_v1.csv"
CACHE = ROOT / ".cache/panderm_base_image_level_v1"
OUTPUT = ROOT / "results/exploratory/panderm_base_image_level_v1"
CHECKPOINT = ROOT / "checkpoints/exploratory/panderm_base_image_level_v1"
BASE = "image_level_weighted_b0_cbam_v1"
STRONG = "image_level_strong_aug_b0_cbam_v1"


def cached_b0(run, size, ids, y):
    with np.load(ROOT / ".cache/exploratory_multires" / f"{run}_{size}.npz") as saved:
        if not np.array_equal(saved["ids"], ids) or not np.array_equal(saved["y"], y):
            raise ValueError("B0 cache does not match exploratory validation")
        return saved["probabilities"]


def main():
    started = time.time()
    strict = pd.read_csv(SPLIT)
    exploratory = pd.read_csv(SPLIT_PATH)
    strict_test = strict.loc[strict.split == "test"]
    if not strict_test.equals(exploratory.loc[exploratory.split == "test"]):
        raise ValueError("Locked strict test assignments changed")
    train = exploratory.loc[exploratory.split == "train"].reset_index(drop=True)
    val = exploratory.loc[exploratory.split == "val"].reset_index(drop=True)
    if len(train) != 7009 or len(val) != 1503 or set(train.image_id) & set(strict_test.image_id) or set(val.image_id) & set(strict_test.image_id):
        raise ValueError("Exploratory development pool changed")
    paths, duplicates = image_paths()
    if duplicates or any(image_id not in paths for image_id in pd.concat([train, val]).image_id):
        raise ValueError("Missing or duplicate development images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_backbone(device)
    x_train = features(model, train, paths, "train", device, CACHE)
    x_val = features(model, val, paths, "val", device, CACHE)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    heads = {
        "panderm_linear": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1500, solver="lbfgs", random_state=42)),
        "panderm_rbf_svc": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=42)),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    ids = val.image_id.to_numpy(dtype=str)
    y = val.label.to_numpy()
    streams = {}
    for name, classifier in heads.items():
        classifier.fit(x_train, train.label.to_numpy())
        if not np.array_equal(classifier[-1].classes_, np.arange(7)):
            raise ValueError("PanDerm class order changed")
        streams[name] = classifier.predict_proba(x_val)
        joblib.dump(classifier, CHECKPOINT / f"{name}.joblib")
        np.savez_compressed(CACHE / f"{name}_val_probabilities.npz", ids=ids, y=y,
                            probabilities=streams[name])
    b0 = {f"{run}_{size}": cached_b0(run, size, ids, y)
          for run in (BASE, STRONG) for size in (224, 384)}
    base_four = sum(b0.values()) / 4
    candidates = {
        "b0_four_control": base_four,
        "panderm_linear": streams["panderm_linear"],
        "panderm_rbf_svc": streams["panderm_rbf_svc"],
        "b0_four_plus_panderm_linear_five_equal": (sum(b0.values()) + streams["panderm_linear"]) / 5,
        "b0_four_plus_panderm_svc_five_equal": (sum(b0.values()) + streams["panderm_rbf_svc"]) / 5,
        "b0_four_plus_two_panderm_six_equal": (sum(b0.values()) + sum(streams.values())) / 6,
    }
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
    shared_lesions = set(train.lesion_id) & set(val.lesion_id)
    write_json(OUTPUT / "protocol.json", {
        "split": SPLIT_PATH.relative_to(ROOT).as_posix(),
        "strict_test_images_evaluated": 0, "train_images": len(train), "validation_images": len(val),
        "train_validation_shared_lesions": len(shared_lesions),
        "validation_images_with_train_lesion": int(val.lesion_id.isin(shared_lesions).sum()),
        "backbone_updated": False, "new_full_network_training_epochs": 0,
        "validation_warning": "Image-level exploratory validation with shared lesions; selected among fixed candidates",
        "runtime_seconds": time.time() - started,
    })


if __name__ == "__main__":
    main()
