"""Phase 5A: train-only metadata preprocessing and validation-only feasibility probe."""
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from torch import nn

from scripts.cache_phase4_features import CACHE, RUNS
from scripts.evaluate_phase4 import read_cache
from src.data import CLASSES, RAW, ROOT
from src.losses import class_weights
from src.metrics import classification_metrics
from src.utils import write_json


OUT = ROOT / "results/phase5_multimodal_probe"
FIELDS = ("age", "sex", "localization")
PAIRS = (("mel", "nv"), ("bkl", "nv"), ("bkl", "mel"), ("akiec", "bcc"))


def save_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def aligned_metadata(train_ids, val_ids):
    # Read only allowed predictors and the join key. Diagnosis and dx_type never enter this frame.
    frame = pd.read_csv(RAW / "HAM10000_metadata.csv", usecols=["image_id", *FIELDS])
    if frame.image_id.duplicated().any():
        raise ValueError("Duplicate metadata image IDs")
    frame = frame.set_index("image_id")
    if any(image_id not in frame.index for image_id in np.concatenate([train_ids, val_ids])):
        raise ValueError("An embedding image ID has no metadata")
    train = frame.loc[train_ids].copy()
    val = frame.loc[val_ids].copy()
    for subset in (train, val):
        subset["age"] = pd.to_numeric(subset["age"], errors="coerce")
        for field in ("sex", "localization"):
            subset[field] = subset[field].fillna("unknown").astype(str).str.strip().str.lower().replace("", "unknown")
    return train, val


def preprocess(train, val):
    median = float(train.age.median())
    train_age = train.age.fillna(median).to_numpy(dtype=np.float32)
    val_age = val.age.fillna(median).to_numpy(dtype=np.float32)
    mean = float(train_age.mean())
    std = float(train_age.std()) or 1.0
    blocks = {"age": ((train_age - mean).reshape(-1, 1) / std,
                      (val_age - mean).reshape(-1, 1) / std)}
    config = {"allowed_predictors": list(FIELDS), "excluded_predictors": ["dx", "dx_type", "lesion_id", "image_id", "split"],
              "age": {"train_median": median, "train_mean_after_imputation": mean, "train_std_after_imputation": std,
                      "missing_train": int(train.age.isna().sum()), "missing_validation": int(val.age.isna().sum())},
              "categorical_unknown_policy": "ignore unknown validation category (all-zero one-hot block)",
              "image_embedding": "1280 post-CBAM/global-pooling features, standardized with train-only mean/std"}
    for field in ("sex", "localization"):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        train_block = encoder.fit_transform(train[[field]])
        val_block = encoder.transform(val[[field]])
        blocks[field] = (train_block, val_block)
        config[field] = {"train_categories": encoder.categories_[0].tolist(),
                         "unknown_validation_rows": int((~val[field].isin(encoder.categories_[0])).sum())}
    write_json(OUT / "preprocessing_config.json", config)
    return blocks


def metric_row(name, y, probabilities, **details):
    metrics = classification_metrics(y, np.asarray(probabilities).argmax(axis=1))
    row = {"method": name, **details, "accuracy": metrics["accuracy"], "macro_precision": metrics["macro_precision"],
           "macro_recall": metrics["macro_recall"], "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"]}
    return row, metrics


def image_metadata_matrix(image_train, image_val, blocks, fields):
    return (np.hstack([image_train, *(blocks[field][0] for field in fields)]).astype(np.float32),
            np.hstack([image_val, *(blocks[field][1] for field in fields)]).astype(np.float32))


def fit_logistic(x_train, y_train, x_val, c=0.3):
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=500)
    model.fit(x_train, y_train)
    if not np.array_equal(model.classes_, np.arange(len(CLASSES))):
        raise ValueError("Logistic class order mismatch")
    return model.predict_proba(x_val)


class SmallMLP(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, len(CLASSES)))

    def forward(self, x):
        return self.net(x)


def fit_mlp(x_train, y_train, x_val, y_val):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallMLP(x_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(np.bincount(y_train, minlength=len(CLASSES))).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    val_x = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    best, best_epoch, stale, best_state = -1.0, 0, 0, None
    history = []
    for epoch in range(1, 41):
        model.train()
        for indices in torch.randperm(len(train_y), device=device).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            probabilities = model(val_x).softmax(1).cpu().numpy()
        row, _ = metric_row("MLP", y_val, probabilities)
        history.append({"epoch": epoch, "validation_accuracy": row["accuracy"], "validation_macro_f1": row["macro_f1"]})
        if row["macro_f1"] > best + 1e-5:
            best, best_epoch, stale = row["macro_f1"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 6:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probabilities = model(val_x).softmax(1).cpu().numpy()
    save_csv(OUT / "mlp_history.csv", history)
    print(f"Small MLP best epoch={best_epoch}, validation macro F1={best:.6f}, device={device}", flush=True)
    return probabilities, best_epoch


def run_probe():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_val, y_val, cnn = read_cache()
    with np.load(CACHE / "weighted_train_embeddings.npz") as saved:
        train_ids = saved["ids"]
        if not np.array_equal(saved["y"], y_train):
            raise ValueError("Train cache alignment mismatch")
    with np.load(CACHE / "weighted_val_embeddings.npz") as saved:
        val_ids = saved["ids"]
        if not np.array_equal(saved["y"], y_val):
            raise ValueError("Validation cache alignment mismatch")
    train_meta, val_meta = aligned_metadata(train_ids, val_ids)
    blocks = preprocess(train_meta, val_meta)
    scaler = StandardScaler()
    image_train = scaler.fit_transform(x_train).astype(np.float32)
    image_val = scaler.transform(x_val).astype(np.float32)
    np.savez_compressed(OUT / "image_scaler_parameters.npz", mean=scaler.mean_, scale=scaler.scale_)

    metadata_train = np.hstack([blocks[field][0] for field in FIELDS]).astype(np.float32)
    metadata_val = np.hstack([blocks[field][1] for field in FIELDS]).astype(np.float32)
    metadata_prob = fit_logistic(metadata_train, y_train, metadata_val)
    metadata_row, metadata_metrics = metric_row("metadata only Logistic Regression C=0.3", y_val, metadata_prob)
    write_json(OUT / "metadata_only_results.json", {**metadata_row, **metadata_metrics})
    print(f"Metadata only: accuracy={metadata_row['accuracy']:.6f} macro F1={metadata_row['macro_f1']:.6f}", flush=True)

    ablations, full_train, full_val = [], None, None
    for name, fields in (("image embeddings only", ()), ("image + age", ("age",)),
                         ("image + sex", ("sex",)), ("image + localization", ("localization",)),
                         ("image + all metadata", FIELDS)):
        train_matrix, val_matrix = image_metadata_matrix(image_train, image_val, blocks, fields)
        probabilities = fit_logistic(train_matrix, y_train, val_matrix)
        row, metrics = metric_row(name, y_val, probabilities, classifier="Logistic Regression", c=0.3,
                                  features="+".join(("image", *fields)), dimension=train_matrix.shape[1])
        ablations.append(row)
        write_json(OUT / f"{name.replace(' ', '_').replace('+', 'plus')}.json", {**row, **metrics})
        print(f"{name}: accuracy={row['accuracy']:.6f} macro F1={row['macro_f1']:.6f}", flush=True)
        if fields == FIELDS:
            full_train, full_val = train_matrix, val_matrix
    save_csv(OUT / "metadata_ablation_results.csv", ablations)

    candidates = []
    candidate_probabilities = {}
    candidate_metrics = {}
    for c in (0.3, 1.0, 3.0):
        probabilities = fit_logistic(full_train, y_train, full_val, c)
        row, metrics = metric_row(f"full Logistic Regression C={c}", y_val, probabilities, family="Logistic Regression", parameter=f"C={c}")
        candidates.append(row)
        candidate_probabilities[row["method"]] = probabilities
        candidate_metrics[row["method"]] = metrics
    for c in (1.0, 3.0):
        model = SVC(C=c, gamma="scale", class_weight="balanced", cache_size=512, random_state=42)
        model.fit(full_train, y_train)
        predicted = model.predict(full_val)
        metrics = classification_metrics(y_val, predicted)
        row = {"method": f"full RBF SVM C={c}", "family": "RBF SVM", "parameter": f"C={c}",
               "accuracy": metrics["accuracy"], "macro_precision": metrics["macro_precision"],
               "macro_recall": metrics["macro_recall"], "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"]}
        candidates.append(row)
        candidate_probabilities[row["method"]] = ("svm", c)
        candidate_metrics[row["method"]] = metrics
        print(f"{row['method']}: accuracy={row['accuracy']:.6f} macro F1={row['macro_f1']:.6f}", flush=True)
    mlp_prob, mlp_epoch = fit_mlp(full_train, y_train, full_val, y_val)
    mlp_row, mlp_metrics = metric_row("full small MLP", y_val, mlp_prob, family="MLP", parameter=f"256-64,dropout=0.3,best_epoch={mlp_epoch}")
    candidates.append(mlp_row)
    candidate_probabilities[mlp_row["method"]] = mlp_prob
    candidate_metrics[mlp_row["method"]] = mlp_metrics
    save_csv(OUT / "multimodal_classifier_results.csv", candidates)
    for row in candidates:
        print(f"{row['method']}: accuracy={row['accuracy']:.6f} macro F1={row['macro_f1']:.6f}", flush=True)
    best_classifier = max(candidates, key=lambda item: (item["macro_f1"], item["accuracy"]))
    best_prob = candidate_probabilities[best_classifier["method"]]
    if isinstance(best_prob, tuple):
        c = best_prob[1]
        calibrated = CalibratedClassifierCV(estimator=SVC(C=c, gamma="scale", class_weight="balanced", cache_size=512, random_state=42),
                                            cv=3, ensemble=False)
        calibrated.fit(full_train, y_train)
        best_prob = calibrated.predict_proba(full_val)
    write_json(OUT / "best_multimodal_classifier.json", {**best_classifier,
        **candidate_metrics[best_classifier["method"]]})

    fusion_rows, fusion_metrics = [], {}
    for protocol in ("normal", "tta"):
        base = cnn[RUNS[0]][protocol]
        for step in range(11):
            cnn_weight = step / 10
            probabilities = cnn_weight * base + (1 - cnn_weight) * best_prob
            name = f"weighted CNN {protocol} + {best_classifier['method']} | CNN={cnn_weight:.1f}"
            row, metrics = metric_row(name, y_val, probabilities, cnn_protocol=protocol,
                                      cnn_weight=cnn_weight, multimodal_weight=1 - cnn_weight,
                                      multimodal_classifier=best_classifier["method"])
            fusion_rows.append(row)
            fusion_metrics[name] = metrics
    save_csv(OUT / "fusion_results.csv", fusion_rows)
    best_fusion_f1 = max(fusion_rows, key=lambda item: (item["macro_f1"], item["accuracy"]))
    best_fusion_accuracy = max(fusion_rows, key=lambda item: (item["accuracy"], item["macro_f1"]))
    write_json(OUT / "best_fusion_macro_f1.json", {**best_fusion_f1, **fusion_metrics[best_fusion_f1["method"]]})
    write_json(OUT / "best_fusion_accuracy.json", {**best_fusion_accuracy, **fusion_metrics[best_fusion_accuracy["method"]]})

    all_new = [*ablations[1:], *candidates, *fusion_rows]
    best_new_f1 = max(all_new, key=lambda item: (item["macro_f1"], item["accuracy"]))
    best_new_accuracy = max(all_new, key=lambda item: (item["accuracy"], item["macro_f1"]))
    if best_new_f1["method"] in fusion_metrics:
        best_metrics = fusion_metrics[best_new_f1["method"]]
    elif best_new_f1["method"] in candidate_metrics:
        best_metrics = candidate_metrics[best_new_f1["method"]]
    else:
        filename = best_new_f1["method"].replace(" ", "_").replace("+", "plus") + ".json"
        best_metrics = json.loads((OUT / filename).read_text(encoding="utf-8"))
    write_json(OUT / "best_confusion_matrix.json", {"method": best_new_f1["method"], "classes": list(CLASSES),
        "confusion_matrix": best_metrics["confusion_matrix"]})
    comparisons = []
    image_best = json.loads((ROOT / "results/phase4/best_ensemble_macro_f1.json").read_text(encoding="utf-8"))
    for name in CLASSES:
        comparisons.append({"class": name, "best_image_only_f1": image_best["per_class"][name]["f1"],
                            "best_multimodal_f1": best_metrics["per_class"][name]["f1"],
                            "delta": best_metrics["per_class"][name]["f1"] - image_best["per_class"][name]["f1"]})
    save_csv(OUT / "per_class_comparison.csv", comparisons)
    pair_rows = []
    for left, right in PAIRS:
        i, j = CLASSES.index(left), CLASSES.index(right)
        pair_rows.append({"pair": f"{left}<->{right}",
                          "best_image_only_errors": image_best["confusion_matrix"][i][j] + image_best["confusion_matrix"][j][i],
                          "best_multimodal_errors": best_metrics["confusion_matrix"][i][j] + best_metrics["confusion_matrix"][j][i]})
    save_csv(OUT / "confusion_pair_comparison.csv", pair_rows)
    image_best_accuracy = json.loads((ROOT / "results/phase4/best_ensemble_accuracy.json").read_text(encoding="utf-8"))["accuracy"]
    summary = {"metadata_only": metadata_row, "ablations": ablations, "best_multimodal_classifier": best_classifier,
               "best_fusion_macro_f1": best_fusion_f1, "best_fusion_accuracy": best_fusion_accuracy,
               "best_new_macro_f1": best_new_f1, "best_new_accuracy": best_new_accuracy,
               "best_image_only_macro_f1": image_best["macro_f1"], "best_image_only_accuracy": image_best_accuracy,
               "runtime_seconds": round(time.monotonic() - started, 1), "test_set_used": False, "cnn_trained": False}
    write_json(OUT / "summary.json", summary)
    print(f"Best multimodal classifier: {best_classifier}", flush=True)
    print(f"Best fusion macro F1: {best_fusion_f1}", flush=True)
    print(f"Best fusion accuracy: {best_fusion_accuracy}", flush=True)
    print(f"Phase 5A complete in {summary['runtime_seconds']} seconds; no CNN training or test access", flush=True)


if __name__ == "__main__":
    run_probe()
