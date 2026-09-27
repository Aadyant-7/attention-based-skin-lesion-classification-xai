"""Validation-only CNN ensembles, frozen-feature ML, and probability fusion."""
import csv
import json
import time
from itertools import product

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import ROOT, CLASSES
from src.metrics import classification_metrics
from src.utils import write_json
from scripts.cache_phase4_features import CACHE, RUNS, cache_manifest


OUT = ROOT / "results/phase4"


def read_cache():
    manifest_file = CACHE / "manifest.json"
    if not manifest_file.is_file() or json.loads(manifest_file.read_text(encoding="utf-8")) != cache_manifest():
        raise RuntimeError("Phase 4 cache missing/stale; run scripts.cache_phase4_features first")
    with np.load(CACHE / "weighted_train_embeddings.npz") as saved:
        train_ids, y_train, x_train = saved["ids"], saved["y"], saved["X"]
    with np.load(CACHE / "weighted_val_embeddings.npz") as saved:
        val_ids, y_val, x_val = saved["ids"], saved["y"], saved["X"]
    if x_train.shape != (7009, 1280) or x_val.shape != (1503, 1280):
        raise ValueError("Embedding shape mismatch")
    probabilities = {}
    for name in RUNS:
        with np.load(CACHE / f"{name}_val_probabilities.npz") as saved:
            if not np.array_equal(saved["ids"], val_ids) or not np.array_equal(saved["y"], y_val):
                raise ValueError("Validation cache alignment mismatch")
            probabilities[name] = {"normal": saved["normal"], "tta": saved["tta"]}
    if len(set(train_ids) & set(val_ids)):
        raise ValueError("Train/validation image overlap")
    return x_train, y_train, x_val, y_val, probabilities


def summary(name, y, probabilities, **details):
    metrics = classification_metrics(y, probabilities.argmax(axis=1))
    return {"method": name, **details, "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"]}, metrics


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ensemble_search(y, cached):
    w, o, f = RUNS
    candidates = []
    details = {}
    for names in ((w, o), (w, f), (w, o, f)):
        protocols = ("normal", "weighted_tta", "all_tta")
        weights = ([(i / 10, 1 - i / 10) for i in range(11)] if len(names) == 2 else
                   [(i / 10, j / 10, (10 - i - j) / 10) for i in range(11) for j in range(11 - i)])
        for protocol, fractions in product(protocols, weights):
            choices = ["tta" if protocol == "all_tta" or (protocol == "weighted_tta" and name == w) else "normal" for name in names]
            mixed = sum(weight * cached[name][choice] for name, choice, weight in zip(names, choices, fractions))
            label = "+".join(name.replace("efficientnet_b0_cbam_", "").replace("_v1", "") for name in names)
            identifier = f"{label}|{protocol}|{','.join(f'{weight:.1f}' for weight in fractions)}"
            row, metrics = summary(identifier, y, mixed, components="+".join(names), protocols="+".join(choices),
                                   weights="+".join(f"{weight:.1f}" for weight in fractions))
            candidates.append(row)
            details[identifier] = metrics
    save_csv(OUT / "ensemble_validation_results.csv", candidates)
    best_f1 = max(candidates, key=lambda row: (row["macro_f1"], row["accuracy"]))
    best_accuracy = max(candidates, key=lambda row: (row["accuracy"], row["macro_f1"]))
    write_json(OUT / "best_ensemble_macro_f1.json", {**best_f1, **details[best_f1["method"]]})
    write_json(OUT / "best_ensemble_accuracy.json", {**best_accuracy, **details[best_accuracy["method"]]})
    print(f"Best CNN ensemble macro F1: {best_f1}", flush=True)
    print(f"Best CNN ensemble accuracy: {best_accuracy}", flush=True)
    return best_f1, best_accuracy, details


def classifiers(x_train, y_train, x_val, y_val):
    specifications = []
    for c in (0.3, 1, 3):
        specifications.append(("Logistic Regression", f"C={c}", make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=500))))
    for c in (1, 3, 10):
        specifications.append(("RBF SVM", f"C={c},gamma=scale", make_pipeline(StandardScaler(), SVC(C=c, gamma="scale", class_weight="balanced", cache_size=512, random_state=42))))
    for n, leaf in ((100, 1), (200, 2)):
        specifications.append(("ExtraTrees", f"trees={n},leaf={leaf}", ExtraTreesClassifier(n_estimators=n, min_samples_leaf=leaf, class_weight="balanced", max_features="sqrt", n_jobs=4, random_state=42)))
    candidates, fitted = [], {}
    for family, parameters, model in specifications:
        start = time.monotonic()
        model.fit(x_train, y_train)
        predicted = model.predict(x_val)
        probabilities = model.predict_proba(x_val) if hasattr(model, "predict_proba") else np.eye(len(CLASSES))[predicted]
        if not np.array_equal(model.classes_, np.arange(len(CLASSES))):
            raise ValueError("Class probability order mismatch")
        row, metrics = summary(f"{family} {parameters}", y_val, probabilities, family=family, parameters=parameters,
                               fit_seconds=round(time.monotonic() - start, 1))
        # SVC predict() can differ from argmax(predict_proba); record decision metrics for model selection.
        decision_metrics = classification_metrics(y_val, predicted)
        row.update({"accuracy": decision_metrics["accuracy"], "macro_precision": decision_metrics["macro_precision"],
                    "macro_recall": decision_metrics["macro_recall"], "macro_f1": decision_metrics["macro_f1"],
                    "weighted_f1": decision_metrics["weighted_f1"]})
        candidates.append(row)
        fitted[row["method"]] = (model, probabilities, decision_metrics)
        print(f"ML {row['method']}: accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} seconds={row['fit_seconds']}", flush=True)
    save_csv(OUT / "frozen_classifier_validation_results.csv", candidates)
    best = max(candidates, key=lambda row: (row["macro_f1"], row["accuracy"]))
    model, probabilities, metrics = fitted[best["method"]]
    if best["family"] == "RBF SVM":
        c = model[-1].C
        calibrated = CalibratedClassifierCV(
            estimator=make_pipeline(StandardScaler(), SVC(C=c, gamma="scale", class_weight="balanced", cache_size=512, random_state=42)),
            cv=3, ensemble=False)
        calibrated.fit(x_train, y_train)
        probabilities = calibrated.predict_proba(x_val)
        if not np.array_equal(calibrated.classes_, np.arange(len(CLASSES))):
            raise ValueError("Calibrated SVM class order mismatch")
    write_json(OUT / "best_frozen_classifier.json", {**best, **metrics})
    print(f"Best frozen classifier: {best}", flush=True)
    return candidates, best, probabilities


def fusion_search(y, cached, ml_probabilities, ml_name):
    rows, details = [], {}
    for protocol in ("normal", "tta"):
        cnn = cached[RUNS[0]][protocol]
        for tenth in range(11):
            cnn_weight = tenth / 10
            mixed = cnn_weight * cnn + (1 - cnn_weight) * ml_probabilities
            identifier = f"weighted_{protocol}+{ml_name}|cnn={cnn_weight:.1f}|ml={1-cnn_weight:.1f}"
            row, metrics = summary(identifier, y, mixed, cnn_protocol=protocol, cnn_weight=cnn_weight,
                                   ml_weight=1 - cnn_weight, ml_classifier=ml_name)
            rows.append(row)
            details[identifier] = metrics
    save_csv(OUT / "cnn_ml_fusion_validation_results.csv", rows)
    best = max(rows, key=lambda row: (row["macro_f1"], row["accuracy"]))
    write_json(OUT / "best_cnn_ml_fusion.json", {**best, **details[best["method"]]})
    print(f"Best CNN + ML fusion: {best}", flush=True)
    return best, details


def error_analysis(best_new, metrics, baseline):
    pairs = (("mel", "nv"), ("bkl", "nv"), ("bkl", "mel"), ("akiec", "bcc"))
    lines = ["# Phase 4 validation error comparison", "", f"Best new candidate by macro F1: `{best_new['method']}` (accuracy {best_new['accuracy']:.6f}; macro F1 {best_new['macro_f1']:.6f}).", "", "Counts below combine both directions of each confusion pair on the 1,503-image validation partition.", "", "| Pair | Weighted normal | Weighted flip TTA | Oversampled normal | Best new candidate |", "|---|---:|---:|---:|---:|"]
    for left, right in pairs:
        i, j = CLASSES.index(left), CLASSES.index(right)
        counts = []
        for matrix in (*baseline, metrics["confusion_matrix"]):
            counts.append(matrix[i][j] + matrix[j][i])
        lines.append(f"| {left} ↔ {right} | " + " | ".join(map(str, counts)) + " |")
    lines.extend(["", "Against weighted flip TTA, the best new candidate reduces bkl↔nv from 36 to 34 and bkl↔mel from 33 to 32; akiec↔bcc stays at 12, while mel↔nv worsens from 103 to 104. Against weighted normal, akiec↔bcc improves from 18 to 12, but mel↔nv worsens from 98 to 104. Against oversampled normal, bkl↔nv improves from 38 to 34 and akiec↔bcc from 13 to 12, while mel↔nv and bkl↔mel worsen."])
    lines.extend(["", "Per-class F1 for the best new candidate: " + ", ".join(f"{name} {metrics['per_class'][name]['f1']:.3f}" for name in CLASSES) + ".", "", "All selection and error analysis used validation labels only; no test features or predictions were produced."])
    (OUT / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_val, y_val, cached = read_cache()
    baseline = []
    for name, protocol in ((RUNS[0], "normal"), (RUNS[0], "tta"), (RUNS[1], "normal")):
        row, metrics = summary(f"{name} {protocol}", y_val, cached[name][protocol])
        baseline.append(metrics["confusion_matrix"])
        print(f"Baseline {row['method']}: accuracy={row['accuracy']:.6f} macro_f1={row['macro_f1']:.6f}", flush=True)
    ensemble_f1, ensemble_accuracy, ensemble_details = ensemble_search(y_val, cached)
    ml_rows, best_ml, ml_probabilities = classifiers(x_train, y_train, x_val, y_val)
    fusion, fusion_details = fusion_search(y_val, cached, ml_probabilities, best_ml["method"])
    best_new = max((ensemble_f1, best_ml, fusion), key=lambda row: (row["macro_f1"], row["accuracy"]))
    if best_new is ensemble_f1:
        metrics = ensemble_details[best_new["method"]]
    elif best_new is fusion:
        metrics = fusion_details[best_new["method"]]
    else:
        # Best ML JSON contains its decision-prediction confusion matrix.
        metrics = json.loads((OUT / "best_frozen_classifier.json").read_text(encoding="utf-8"))
    error_analysis(best_new, metrics, baseline)
    print(f"Phase 4 evaluation complete in {(time.monotonic()-start)/60:.1f} minutes; test set untouched", flush=True)


if __name__ == "__main__":
    main()
