"""Two cheap, fixed heads on cached strict PanDerm Base embeddings."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import CLASSES, ROOT, SPLIT
from src.metrics import classification_metrics
from src.utils import write_json


CACHE = ROOT / ".cache/panderm_base_strict_lp_v1"
OUTPUT = ROOT / "results/accuracy_exploration/panderm_base_strict_heads_v1"
HEADS = ROOT / "checkpoints/accuracy_exploration/panderm_base_strict_heads_v1"


def main():
    frame = pd.read_csv(SPLIT)
    train = frame.loc[frame.split == "train"].reset_index(drop=True)
    val = frame.loc[frame.split == "val"].reset_index(drop=True)
    arrays = {}
    for name, rows in (("train", train), ("val", val)):
        with np.load(CACHE / f"{name}.npz") as saved:
            if not np.array_equal(saved["ids"], rows.image_id.to_numpy(dtype=str)) or not np.array_equal(saved["y"], rows.label.to_numpy()):
                raise ValueError("Feature cache does not match strict split")
            arrays[name] = saved["features"]
    baseline = {}
    for run in ("efficientnet_b0_cbam_weighted_v1", "efficientnet_b0_cbam_oversampled_v1"):
        with np.load(ROOT / ".cache/phase4" / f"{run}_val_probabilities.npz") as saved:
            if not np.array_equal(saved["ids"], val.image_id.to_numpy(dtype=str)) or not np.array_equal(saved["y"], val.label.to_numpy()):
                raise ValueError("B0 cache does not match strict split")
            baseline[run] = saved["normal"]
    heads = {
        "standardized_logreg_c1": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1500, solver="lbfgs", random_state=42)),
        "standardized_rbf_svc_c10": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=42)),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    HEADS.mkdir(parents=True, exist_ok=True)
    for name, classifier in heads.items():
        started = time.time()
        classifier.fit(arrays["train"], train.label.to_numpy())
        classes = classifier[-1].classes_
        if not np.array_equal(classes, np.arange(len(CLASSES))):
            raise ValueError("Class order changed")
        probabilities = classifier.predict_proba(arrays["val"])
        candidate = {
            name: probabilities,
            name + "_plus_weighted_b0_equal": (probabilities + baseline["efficientnet_b0_cbam_weighted_v1"]) / 2,
            name + "_plus_oversampled_b0_equal": (probabilities + baseline["efficientnet_b0_cbam_oversampled_v1"]) / 2,
        }
        for method, probs in candidate.items():
            metrics = classification_metrics(val.label.to_numpy(), probs.argmax(axis=1))
            write_json(OUTPUT / f"{method}.json", metrics)
            print(json.dumps({"method": method, "accuracy": metrics["accuracy"],
                              "macro_f1": metrics["macro_f1"]}), flush=True)
        joblib.dump(classifier, HEADS / f"{name}.joblib")
        np.savez_compressed(CACHE / f"{name}_val_probabilities.npz",
                            ids=val.image_id.to_numpy(dtype=str), y=val.label.to_numpy(),
                            probabilities=probabilities)
        write_json(OUTPUT / f"{name}_protocol.json", {
            "model": "frozen PanDerm Base", "head": name, "strict_train_images": len(train),
            "strict_validation_images": len(val), "test_images_evaluated": 0,
            "validation_warning": "Candidates inspected on strict validation; locked test remains untouched",
            "runtime_seconds": time.time() - started})


if __name__ == "__main__":
    main()
