"""Validation-only MC dropout on the fixed weighted checkpoint, plus a resize diagnostic."""
import csv
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms

from scripts.cache_phase4_features import CACHE, RUNS, cache_manifest
from src.data import CLASSES, MEAN, STD, ROOT, SPLIT, LesionDataset, image_paths
from src.metrics import classification_metrics
from src.models import EfficientNetCBAM
from src.train import loader
from src.utils import write_json


OUT = ROOT / "results/phase5b_inference_audit"
LOCAL_CACHE = ROOT / ".cache/phase5b"
PASSES = (10, 20, 30)


def summarize(name, y, probabilities, **details):
    metrics = classification_metrics(y, np.asarray(probabilities).argmax(1))
    return ({"method": name, **details, "accuracy": metrics["accuracy"],
             "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"],
             "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"]}, metrics)


def flipped_embeddings(model, val_rows, paths, ids, device):
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    path = LOCAL_CACHE / "weighted_flipped_val_embeddings.npz"
    stamp = {"split_sha256": cache_manifest()["split_sha256"],
             "checkpoint": cache_manifest()["checkpoints"][RUNS[0]]}
    stamp_path = LOCAL_CACHE / "weighted_flipped_val_embeddings.json"
    if path.is_file() and stamp_path.is_file() and json.loads(stamp_path.read_text(encoding="utf-8")) == stamp:
        with np.load(path) as saved:
            if np.array_equal(saved["ids"], ids) and saved["X"].shape == (1503, 1280):
                print("Reused flipped validation embedding cache", flush=True)
                return saved["X"]
    batches = loader(LesionDataset(val_rows, paths, training=False, size=224), 64, 2)
    vectors = []
    with torch.inference_mode():
        for images, _ in batches:
            images = torch.flip(images.to(device, non_blocking=True), dims=(-1,))
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                features = model.pool(model.cbam(model.features(images))).flatten(1)
            vectors.append(features.float().cpu().numpy())
    result = np.concatenate(vectors)
    if result.shape != (1503, 1280):
        raise ValueError("Unexpected flipped embedding shape")
    np.savez_compressed(path, ids=ids.astype(str), X=result)
    stamp_path.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    print("Cached flipped validation embeddings; test images not loaded", flush=True)
    return result


def mc_predictions(model, original, flipped, device):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dropouts = [module for module in model.modules() if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d))]
    if not dropouts:
        raise ValueError("No Dropout layers found")
    for module in dropouts:
        module.train()
    if any(module.training for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)):
        raise AssertionError("BatchNorm must remain in eval mode")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    collected = {(passes, flip): [] for passes in PASSES for flip in (False, True)}
    head = model.classifier
    with torch.inference_mode():
        for offset in range(0, len(original), 256):
            x = torch.as_tensor(original[offset:offset + 256], dtype=torch.float32, device=device)
            flipped_x = torch.as_tensor(flipped[offset:offset + 256], dtype=torch.float32, device=device)
            sum_normal = torch.zeros((len(x), len(CLASSES)), dtype=torch.float32, device=device)
            sum_flipped = torch.zeros_like(sum_normal)
            for index in range(1, max(PASSES) + 1):
                with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    sum_normal += head(x).float().softmax(1)
                    sum_flipped += head(flipped_x).float().softmax(1)
                if index in PASSES:
                    collected[(index, False)].append((sum_normal / index).cpu().numpy())
                    collected[(index, True)].append(((sum_normal + sum_flipped) / (2 * index)).cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in collected.items()}, len(dropouts)


def aspect_ratio_diagnostic(model, val_rows, paths, device):
    model.eval()
    dataset = LesionDataset(val_rows, paths, training=False, size=224)
    dataset.transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    batches = loader(dataset, 64, 2)
    probabilities = []
    with torch.inference_mode():
        for images, _ in batches:
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                probabilities.append(model(images.to(device, non_blocking=True)).float().softmax(1).cpu().numpy())
    return np.concatenate(probabilities)


def save_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    start = time.monotonic()
    if not SPLIT.is_file():
        raise FileNotFoundError(f"Saved split is missing: {SPLIT}")
    with np.load(CACHE / "weighted_val_embeddings.npz") as saved:
        ids, y, original = saved["ids"], saved["y"], saved["X"]
    with np.load(CACHE / f"{RUNS[0]}_val_probabilities.npz") as saved:
        if not np.array_equal(saved["ids"], ids) or not np.array_equal(saved["y"], y):
            raise ValueError("Cached validation probabilities are not aligned")
        baseline_prob = {"weighted normal": saved["normal"], "weighted flip TTA": saved["tta"]}
    assignments = pd.read_csv(SPLIT)
    validation = assignments.loc[assignments.split == "val"].reset_index(drop=True)
    if len(validation) != 1503 or not np.array_equal(validation.image_id.to_numpy(dtype=str), ids):
        raise ValueError("Cached embeddings do not match fixed validation rows")
    paths, duplicates = image_paths()
    if duplicates or any(image_id not in paths for image_id in ids):
        raise ValueError("Validation image mapping incomplete")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ROOT / "checkpoints" / RUNS[0] / "best.pt", map_location=device, weights_only=False)
    model = EfficientNetCBAM(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # Confirm that the cached deterministic embeddings reproduce the baseline head predictions.
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
        cached_head = np.concatenate([model.classifier(torch.as_tensor(original[i:i + 256], device=device)).float().softmax(1).cpu().numpy()
                                      for i in range(0, len(original), 256)])
    if not np.array_equal(cached_head.argmax(1), baseline_prob["weighted normal"].argmax(1)):
        raise ValueError("Cached embeddings do not reproduce weighted normal predictions")
    flipped = flipped_embeddings(model, validation, paths, ids, device)
    predictions, dropout_count = mc_predictions(model, original, flipped, device)
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_rows, baseline_metrics = {}, {}
    for name, probabilities in baseline_prob.items():
        baseline_rows[name], baseline_metrics[name] = summarize(name, y, probabilities)
    rows, detail = [], {}
    for passes in PASSES:
        for with_flip in (False, True):
            name = f"MC dropout {passes} passes" + (" + flip TTA" if with_flip else "")
            row, metrics = summarize(name, y, predictions[(passes, with_flip)], passes=passes,
                                     flip_tta=with_flip)
            rows.append(row)
            detail[name] = metrics
            write_json(OUT / f"mc_{passes}_{'flip' if with_flip else 'normal'}.json", {**row, **metrics})
            print(f"{name}: accuracy={row['accuracy']:.6f} macro_f1={row['macro_f1']:.6f}", flush=True)
    save_csv(OUT / "mc_dropout_results.csv", rows)
    best = max(rows, key=lambda row: (row["macro_f1"], row["accuracy"]))
    write_json(OUT / "best_confusion_matrix.json", {"method": best["method"], "classes": list(CLASSES),
        "confusion_matrix": detail[best["method"]]["confusion_matrix"]})
    phase4 = json.loads((ROOT / "results/phase4/best_ensemble_macro_f1.json").read_text(encoding="utf-8"))
    comparisons = []
    for name in CLASSES:
        comparisons.append({"class": name, "weighted_normal_f1": baseline_metrics["weighted normal"]["per_class"][name]["f1"],
                            "weighted_flip_tta_f1": baseline_metrics["weighted flip TTA"]["per_class"][name]["f1"],
                            "best_phase4_ensemble_f1": phase4["per_class"][name]["f1"],
                            "best_mc_dropout_f1": detail[best["method"]]["per_class"][name]["f1"]})
    save_csv(OUT / "per_class_comparison.csv", comparisons)
    model.eval()  # Disable Dropout again for the deterministic preprocessing diagnostic.
    aspect_prob = aspect_ratio_diagnostic(model, validation, paths, device)
    aspect_row, aspect_metrics = summarize("aspect-preserving resize + center crop diagnostic", y, aspect_prob)
    write_json(OUT / "aspect_ratio_diagnostic.json", {**aspect_row, **aspect_metrics})
    summary = {"best_mc_dropout": best, "dropout_layers_enabled": dropout_count,
               "aspect_ratio_diagnostic": aspect_row, "runtime_seconds": round(time.monotonic() - start, 1),
               "cnn_trained": False, "test_set_used": False}
    write_json(OUT / "summary.json", summary)
    print(f"Aspect-ratio diagnostic: accuracy={aspect_row['accuracy']:.6f} macro_f1={aspect_row['macro_f1']:.6f}", flush=True)
    print(f"Phase 5B inference complete in {summary['runtime_seconds']} seconds; no training or test access", flush=True)


if __name__ == "__main__":
    main()
