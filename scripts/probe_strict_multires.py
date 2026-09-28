"""Frozen multiresolution inference on the locked strict validation partition."""
import csv
import json

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.data import ROOT, SPLIT, LesionDataset, image_paths
from src.metrics import classification_metrics
from src.models import EfficientNetCBAM
from src.train import loader
from src.utils import write_json


RUNS = ("efficientnet_b0_cbam_weighted_v1", "efficientnet_b0_cbam_oversampled_v1")
OUTPUT = ROOT / "results/strict_multires_probe"
CACHE = ROOT / ".cache/strict_multires"


def infer_384(run, rows, paths):
    cache_path = CACHE / f"{run}_384.npz"
    if cache_path.exists():
        with np.load(cache_path) as data:
            if np.array_equal(data["ids"], rows.image_id.to_numpy(dtype=str)):
                return data["probabilities"]
        raise ValueError("Stale 384 cache ordering")
    saved = torch.load(ROOT / "checkpoints" / run / "best.pt", map_location="cuda", weights_only=False)
    model = EfficientNetCBAM(pretrained=False).cuda().eval()
    model.load_state_dict(saved["model"])
    vectors = []
    batches = loader(LesionDataset(rows, paths, size=384), 24, 2)
    with torch.inference_mode():
        for x, _ in batches:
            x = x.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda"):
                logits = model(x)
            vectors.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    probabilities = np.concatenate(vectors)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ids=rows.image_id.to_numpy(dtype=str),
                        y=rows.label.to_numpy(), probabilities=probabilities)
    return probabilities


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    split = pd.read_csv(SPLIT)
    rows = split.loc[split.split == "val"].reset_index(drop=True)
    assert len(rows) == 1503 and rows.image_id.is_unique
    assert not set(rows.lesion_id) & set(split.loc[split.split == "test", "lesion_id"])
    paths, duplicates = image_paths()
    if duplicates or any(x not in paths for x in rows.image_id):
        raise ValueError("Missing or duplicate validation paths")
    streams = {}
    controls = ((.8456420492348636, .7568732673677362),
                (.852960745176314, .732131726806851))
    for run, expected in zip(RUNS, controls):
        with np.load(ROOT / ".cache/phase4" / f"{run}_val_probabilities.npz") as data:
            if not np.array_equal(data["ids"], rows.image_id.to_numpy(dtype=str)) or not np.array_equal(data["y"], rows.label.to_numpy()):
                raise ValueError("Phase4 cache ordering differs from strict validation")
            normal, tta = data["normal"], data["tta"]
        measured = classification_metrics(rows.label, normal.argmax(axis=1))
        if abs(measured["accuracy"]-expected[0]) > 1e-12 or abs(measured["macro_f1"]-expected[1]) > 1e-12:
            raise AssertionError("Strict 224-pixel control failed")
        streams[run+"_224"] = normal
        streams[run+"_tta"] = tta
        streams[run+"_384"] = infer_384(run, rows, paths)
        print(f"Cached strict 384 inference for {run}", flush=True)
    w, o = RUNS
    candidates = {
        "weighted_224": {w+"_224": 1.0},
        "oversampled_224": {o+"_224": 1.0},
        "weighted_224_384": {w+"_224": .5, w+"_384": .5},
        "oversampled_224_384": {o+"_224": .5, o+"_384": .5},
        "two_models_two_resolutions": {w+"_224": .25, w+"_384": .25, o+"_224": .25, o+"_384": .25},
        "two_models_tta_and_384": {w+"_tta": .25, w+"_384": .25, o+"_tta": .25, o+"_384": .25},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, weights in candidates.items():
        probs = sum(value*streams[key] for key,value in weights.items())
        metrics = classification_metrics(rows.label, probs.argmax(axis=1))
        summary = {"method": name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                   "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"]}
        summaries.append(summary)
        write_json(OUTPUT / f"{name}.json", {"weights": weights, **metrics})
        print(json.dumps(summary), flush=True)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(OUTPUT / "protocol.json", {"strict_validation_images": len(rows), "strict_test_evaluated": False,
        "model_weights_updated": False, "source_checkpoints": [f"checkpoints/{run}/best.pt" for run in RUNS],
        "selection_warning": "Multiple inference variants scored on the same strict validation split"})


if __name__ == "__main__":
    main()
