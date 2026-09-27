"""Training-prior and cost-weight corrections on cached validation probabilities only."""
import csv
import json
import time

import numpy as np

from scripts.cache_phase4_features import CACHE, RUNS, cache_manifest
from src.data import CLASSES, ROOT, SPLIT
from src.losses import class_weights
from src.metrics import classification_metrics
from src.utils import write_json


OUT = ROOT / "results/phase5c_prior_correction"
MODEL_NAMES = {RUNS[0]: "weighted", RUNS[1]: "oversampled"}
GRID = tuple(step / 10 for step in range(11))
PAIRS = (("mel", "nv"), ("bkl", "nv"), ("bkl", "mel"), ("akiec", "bcc"))


def train_counts():
    if not SPLIT.is_file():
        raise FileNotFoundError(f"Saved split is missing: {SPLIT}")
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    with SPLIT.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] == "train":
                counts[int(row["label"])] += 1
    if counts.sum() != 7009 or np.any(counts == 0):
        raise ValueError(f"Unexpected original training counts: {counts}")
    return counts


def cached_probabilities():
    manifest_file = CACHE / "manifest.json"
    if not manifest_file.is_file() or json.loads(manifest_file.read_text(encoding="utf-8")) != cache_manifest():
        raise RuntimeError("Phase 4 probability cache is missing or stale")
    streams, ids, labels = {}, None, None
    for run in MODEL_NAMES:
        with np.load(CACHE / f"{run}_val_probabilities.npz") as saved:
            if ids is None:
                ids, labels = saved["ids"], saved["y"]
            elif not np.array_equal(saved["ids"], ids) or not np.array_equal(saved["y"], labels):
                raise ValueError("Validation probability caches are misaligned")
            for protocol in ("normal", "tta"):
                probabilities = saved[protocol].astype(np.float64)
                if probabilities.shape != (1503, len(CLASSES)) or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
                    raise ValueError("Invalid cached validation probabilities")
                streams[(MODEL_NAMES[run], protocol)] = probabilities
    return labels, streams


def corrected(probabilities, log_shift, strength):
    adjusted = np.log(np.clip(probabilities, 1e-30, 1)) + strength * log_shift
    adjusted -= adjusted.max(axis=1, keepdims=True)
    result = np.exp(adjusted)
    return result / result.sum(axis=1, keepdims=True)


def evaluate(name, labels, probabilities, model, protocol, strength, correction_type):
    metrics = classification_metrics(labels, probabilities.argmax(1))
    row = {"method": name, "model": model, "protocol": protocol, "correction_type": correction_type,
           "lambda": strength, "accuracy": metrics["accuracy"], "macro_precision": metrics["macro_precision"],
           "macro_recall": metrics["macro_recall"], "macro_f1": metrics["macro_f1"],
           "weighted_f1": metrics["weighted_f1"],
           **{f"f1_{cls}": metrics["per_class"][cls]["f1"] for cls in CLASSES},
           "confusion_matrix_json": json.dumps(metrics["confusion_matrix"], separators=(",", ":"))}
    return row, metrics


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pair_counts(matrix):
    return {f"{a}<->{b}": matrix[CLASSES.index(a)][CLASSES.index(b)] + matrix[CLASSES.index(b)][CLASSES.index(a)]
            for a, b in PAIRS}


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    counts = train_counts()
    target_prior = counts / counts.sum()
    uniform_source = np.full(len(CLASSES), 1 / len(CLASSES))
    weights = class_weights(counts).numpy().astype(np.float64)
    write_json(OUT / "training_prior_and_weights.json", {"train_counts": dict(zip(CLASSES, counts.tolist())),
        "target_prior": dict(zip(CLASSES, target_prior.tolist())),
        "oversampled_source_prior": dict(zip(CLASSES, uniform_source.tolist())),
        "weighted_ce_class_weights": dict(zip(CLASSES, weights.tolist())),
        "oversampled_log_shift": "log(target_prior) - log(uniform_source_prior)",
        "weighted_log_shift": "-log(weight); exploratory cost correction, not exact Bayesian calibration"})
    labels, streams = cached_probabilities()
    shifts = {"oversampled": np.log(target_prior) - np.log(uniform_source), "weighted": -np.log(weights)}
    rows, details, probabilities_by_key = [], {}, {}
    for (model, protocol), probabilities in streams.items():
        for strength in GRID:
            key = (model, protocol, strength)
            name = f"{model}_{protocol}_lambda_{strength:.1f}"
            adjusted = corrected(probabilities, shifts[model], strength)
            row, metrics = evaluate(name, labels, adjusted, model, protocol, strength,
                                    "prior_shift" if model == "oversampled" else "cost_weight_exploratory")
            rows.append(row)
            details[key] = metrics
            probabilities_by_key[key] = adjusted
    save_csv(OUT / "correction_grid.csv", rows)
    best_oversampled_f1 = max((row for row in rows if row["model"] == "oversampled" and row["lambda"] > 0),
                              key=lambda row: (row["macro_f1"], row["accuracy"]))
    best_oversampled_accuracy = max((row for row in rows if row["model"] == "oversampled" and row["lambda"] > 0),
                                    key=lambda row: (row["accuracy"], row["macro_f1"]))
    best_weighted_f1 = max((row for row in rows if row["model"] == "weighted" and row["lambda"] > 0),
                           key=lambda row: (row["macro_f1"], row["accuracy"]))
    best_weighted_accuracy = max((row for row in rows if row["model"] == "weighted" and row["lambda"] > 0),
                                 key=lambda row: (row["accuracy"], row["macro_f1"]))
    best_corrected_accuracy = max((row for row in rows if row["lambda"] > 0), key=lambda row: (row["accuracy"], row["macro_f1"]))
    best_corrected_f1 = max((row for row in rows if row["lambda"] > 0), key=lambda row: (row["macro_f1"], row["accuracy"]))
    for label, row in (("best_oversampled_macro_f1", best_oversampled_f1),
                       ("best_oversampled_accuracy", best_oversampled_accuracy),
                       ("best_weighted_macro_f1", best_weighted_f1),
                       ("best_weighted_accuracy", best_weighted_accuracy)):
        key = (row["model"], row["protocol"], row["lambda"])
        write_json(OUT / f"{label}.json", {**row, **details[key]})
    for label, row in (("best_accuracy_confusion_matrix", best_corrected_accuracy),
                       ("best_macro_f1_confusion_matrix", best_corrected_f1)):
        key = (row["model"], row["protocol"], row["lambda"])
        write_json(OUT / f"{label}.json", {"method": row["method"], "classes": list(CLASSES),
            "confusion_matrix": details[key]["confusion_matrix"]})

    comparisons = []
    for label, candidate in (("best_corrected_accuracy", best_corrected_accuracy),
                             ("best_corrected_macro_f1", best_corrected_f1)):
        key = (candidate["model"], candidate["protocol"], candidate["lambda"])
        baseline = details[(candidate["model"], candidate["protocol"], 0.0)]
        metrics = details[key]
        for cls in CLASSES:
            comparisons.append({"candidate": label, "model": candidate["model"], "protocol": candidate["protocol"],
                                "lambda": candidate["lambda"], "class": cls,
                                "baseline_f1": baseline["per_class"][cls]["f1"],
                                "corrected_f1": metrics["per_class"][cls]["f1"],
                                "delta_f1": metrics["per_class"][cls]["f1"] - baseline["per_class"][cls]["f1"]})
    save_csv(OUT / "per_class_comparison.csv", comparisons)

    # A combination is warranted only for a clear >=0.005 gain over its own uncorrected stream.
    clear = [row for row in rows if row["lambda"] > 0 and
             (row["accuracy"] - details[(row["model"], row["protocol"], 0.0)]["accuracy"] >= 0.005 or
              row["macro_f1"] - details[(row["model"], row["protocol"], 0.0)]["macro_f1"] >= 0.005)]
    combination_rows = []
    if clear:
        with np.load(CACHE / f"{RUNS[2]}_val_probabilities.npz") as saved:
            focal_normal = saved["normal"].astype(np.float64)
            if not np.array_equal(saved["y"], labels):
                raise ValueError("Focal validation probabilities misaligned")
        # Reconstruct the saved Phase 4 macro-F1 ensemble: 90% weighted TTA + 10% focal normal.
        reference = 0.9 * streams[("weighted", "tta")] + 0.1 * focal_normal
        selected = {candidate["method"]: candidate for candidate in
                    (max(clear, key=lambda row: (row["accuracy"], row["macro_f1"])),
                     max(clear, key=lambda row: (row["macro_f1"], row["accuracy"]))) }
        for candidate in selected.values():
            key = (candidate["model"], candidate["protocol"], candidate["lambda"])
            for corrected_weight in (0.25, 0.5, 0.75):
                mixed = corrected_weight * probabilities_by_key[key] + (1 - corrected_weight) * reference
                name = f"phase4_ensemble+{candidate['method']}_weight_{corrected_weight:.2f}"
                row, metrics = evaluate(name, labels, mixed, candidate["model"], candidate["protocol"], candidate["lambda"],
                                        "small_combination")
                row["corrected_weight"] = corrected_weight
                combination_rows.append(row)
        save_csv(OUT / "combination_results.csv", combination_rows)
    elif (OUT / "combination_results.csv").exists():
        (OUT / "combination_results.csv").unlink()

    overall_reference = json.loads((ROOT / "results/phase4/best_ensemble_macro_f1.json").read_text(encoding="utf-8"))
    max_reference_accuracy = json.loads((ROOT / "results/phase4/best_ensemble_accuracy.json").read_text(encoding="utf-8"))["accuracy"]
    best_grid_acc = max(rows, key=lambda row: (row["accuracy"], row["macro_f1"]))
    best_grid_f1 = max(rows, key=lambda row: (row["macro_f1"], row["accuracy"]))
    best_combination_acc = max(combination_rows, key=lambda row: (row["accuracy"], row["macro_f1"])) if combination_rows else None
    best_combination_f1 = max(combination_rows, key=lambda row: (row["macro_f1"], row["accuracy"])) if combination_rows else None
    summary = {"best_oversampled_macro_f1": best_oversampled_f1, "best_oversampled_accuracy": best_oversampled_accuracy,
               "best_weighted_macro_f1": best_weighted_f1, "best_weighted_accuracy": best_weighted_accuracy,
               "best_corrected_macro_f1": best_corrected_f1, "best_corrected_accuracy": best_corrected_accuracy,
               "best_grid_macro_f1": best_grid_f1, "best_grid_accuracy": best_grid_acc,
               "best_combination_macro_f1": best_combination_f1, "best_combination_accuracy": best_combination_acc,
               "reference_best_image_only_macro_f1": overall_reference["macro_f1"],
               "reference_best_image_only_accuracy": max_reference_accuracy,
               "runtime_seconds": round(time.monotonic() - start, 2), "test_set_used": False, "model_trained": False}
    write_json(OUT / "summary.json", summary)
    lines = ["# Phase 5C validation-only prior and cost correction", "",
        "Target prior was computed exclusively from the original 7,009 **training** split rows. Oversampled source prior was exactly uniform. Weighted-CE correction is an exploratory cost-weight adjustment, not exact Bayesian calibration.", "",
        "For both normal and horizontal-flip-TTA probability streams, the saved grid tests lambda = 0.0, 0.1, ..., 1.0. Corrected probabilities are normalized after adding the log shift. Full per-class F1 and confusion matrices for all 44 grid candidates are in `correction_grid.csv`.", "",
        f"Best corrected accuracy: `{best_corrected_accuracy['method']}` = {best_corrected_accuracy['accuracy']:.6f} accuracy, {best_corrected_accuracy['macro_f1']:.6f} macro F1.",
        f"Best corrected macro F1: `{best_corrected_f1['method']}` = {best_corrected_f1['accuracy']:.6f} accuracy, {best_corrected_f1['macro_f1']:.6f} macro F1.",
        f"Existing image-only references: {max_reference_accuracy:.6f} best accuracy; {overall_reference['macro_f1']:.6f} best macro F1.", ""]
    for label, candidate in (("Best corrected accuracy", best_corrected_accuracy), ("Best corrected macro F1", best_corrected_f1)):
        key = (candidate["model"], candidate["protocol"], candidate["lambda"])
        base = details[(candidate["model"], candidate["protocol"], 0.0)]
        corrected_metrics = details[key]
        lines.append(f"{label} versus its own uncorrected stream: NV F1 {base['per_class']['nv']['f1']:.3f} → {corrected_metrics['per_class']['nv']['f1']:.3f}; MEL F1 {base['per_class']['mel']['f1']:.3f} → {corrected_metrics['per_class']['mel']['f1']:.3f}; mel↔nv errors {pair_counts(base['confusion_matrix'])['mel<->nv']} → {pair_counts(corrected_metrics['confusion_matrix'])['mel<->nv']}.")
    accuracy_key = (best_corrected_accuracy["model"], best_corrected_accuracy["protocol"], best_corrected_accuracy["lambda"])
    accuracy_base = details[(best_corrected_accuracy["model"], best_corrected_accuracy["protocol"], 0.0)]
    base_matrix = accuracy_base["confusion_matrix"]
    corrected_matrix = details[accuracy_key]["confusion_matrix"]
    lines.extend(["", f"The best corrected-accuracy stream changes predicted NV count from {sum(row[CLASSES.index('nv')] for row in base_matrix)} to {sum(row[CLASSES.index('nv')] for row in corrected_matrix)}; correct NV predictions {base_matrix[CLASSES.index('nv')][CLASSES.index('nv')]} → {corrected_matrix[CLASSES.index('nv')][CLASSES.index('nv')]}, correct MEL predictions {base_matrix[CLASSES.index('mel')][CLASSES.index('mel')]} → {corrected_matrix[CLASSES.index('mel')][CLASSES.index('mel')]}. MEL→NV rises {base_matrix[CLASSES.index('mel')][CLASSES.index('nv')]} → {corrected_matrix[CLASSES.index('mel')][CLASSES.index('nv')]}, while NV→MEL falls {base_matrix[CLASSES.index('nv')][CLASSES.index('mel')]} → {corrected_matrix[CLASSES.index('nv')][CLASSES.index('mel')]}. The small accuracy gain is driven by majority-class predictions, not balanced class improvement."])
    if best_combination_f1:
        lines.append(f"Optional three-weight combination with the existing Phase 4 ensemble did not help: its best macro F1 was {best_combination_f1['macro_f1']:.6f} (accuracy {best_combination_f1['accuracy']:.6f}).")
    lines.extend(["", "The full correction is a theoretical diagnostic. Coarse lambda selection uses validation labels and is exploratory; it is not a held-out estimate. No model was trained and no test data were used."])
    (OUT / "conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Best oversampled correction by macro F1: {best_oversampled_f1['method']} accuracy={best_oversampled_f1['accuracy']:.6f} macro_f1={best_oversampled_f1['macro_f1']:.6f}", flush=True)
    print(f"Best weighted correction by macro F1: {best_weighted_f1['method']} accuracy={best_weighted_f1['accuracy']:.6f} macro_f1={best_weighted_f1['macro_f1']:.6f}", flush=True)
    print(f"Best corrected accuracy: {best_corrected_accuracy['method']} accuracy={best_corrected_accuracy['accuracy']:.6f} macro_f1={best_corrected_accuracy['macro_f1']:.6f}", flush=True)
    print(f"Best corrected macro F1: {best_corrected_f1['method']} accuracy={best_corrected_f1['accuracy']:.6f} macro_f1={best_corrected_f1['macro_f1']:.6f}", flush=True)
    print(f"Phase 5C completed in {summary['runtime_seconds']} seconds; no training or test access", flush=True)


if __name__ == "__main__":
    main()
