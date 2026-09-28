"""Frozen-checkpoint resolution probe on exploratory validation images only."""
import argparse
import csv
import json

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.data import ROOT, LesionDataset, image_paths
from src.metrics import classification_metrics
from src.models import EfficientNetCBAM
from src.train import loader
from src.utils import write_json


RUNS = {
    "image_level_weighted_b0_cbam_v1": (.8622754491017964, .7807634025293032),
    "image_level_strong_aug_b0_cbam_v1": (.852960745176314, .7657417116155746),
}
SPLIT = ROOT / "data/splits/exploratory/image_level_dev_v1.csv"
OUTPUT_ROOT = ROOT / "results/exploratory/resolution_probe"


def probabilities(model, rows, paths, size, device):
    batches = loader(LesionDataset(rows, paths, size=size), 64 if size == 224 else 24, 2)
    vectors = []
    with torch.inference_mode():
        for x, _ in batches:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(x)
            vectors.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS, default="image_level_weighted_b0_cbam_v1")
    args = parser.parse_args()
    output = OUTPUT_ROOT / args.run
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this probe")
    assignments = pd.read_csv(SPLIT)
    rows = assignments.loc[assignments.split == "val"].reset_index(drop=True)
    assert len(rows) == 1503 and rows.image_id.is_unique
    strict = pd.read_csv(ROOT / "data/splits/split_assignments.csv")
    strict_test = set(strict.loc[strict.split == "test", "image_id"])
    assert not (set(rows.image_id) & strict_test)
    paths, duplicates = image_paths()
    if duplicates or any(x not in paths for x in rows.image_id):
        raise ValueError("Validation image paths missing or duplicated")
    checkpoint = ROOT / "checkpoints/exploratory" / args.run / "best.pt"
    saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
    assert saved["best_epoch"] == (19 if args.run == "image_level_weighted_b0_cbam_v1" else 18)
    model = EfficientNetCBAM(pretrained=False).cuda().eval()
    model.load_state_dict(saved["model"])
    truth = rows.label.to_numpy()
    streams = {}
    summaries = []
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache/exploratory_multires"
    cache.mkdir(parents=True, exist_ok=True)
    for size in (224, 320, 384):
        streams[size] = probabilities(model, rows, paths, size, torch.device("cuda"))
        metrics = classification_metrics(truth, streams[size].argmax(axis=1))
        print(json.dumps({"control_size": size, "accuracy": metrics["accuracy"],
                          "macro_f1": metrics["macro_f1"]}), flush=True)
        if size == 224 and (abs(metrics["accuracy"] - RUNS[args.run][0]) > 1e-12 or
                            abs(metrics["macro_f1"] - RUNS[args.run][1]) > 1e-12):
            raise AssertionError("224-pixel control does not reproduce saved best metrics")
        np.savez_compressed(cache / f"{args.run}_{size}.npz", ids=rows.image_id.to_numpy(dtype=str),
                            y=truth, probabilities=streams[size])
        summary = {"method": f"frozen_{size}", "accuracy": metrics["accuracy"],
                   "macro_f1": metrics["macro_f1"], "macro_precision": metrics["macro_precision"],
                   "macro_recall": metrics["macro_recall"]}
        summaries.append(summary)
        write_json(output / f"frozen_{size}.json", metrics)
        print(json.dumps(summary), flush=True)
    mixed = (streams[224] + streams[384]) / 2
    metrics = classification_metrics(truth, mixed.argmax(axis=1))
    summary = {"method": "frozen_224_384_equal_average", "accuracy": metrics["accuracy"],
               "macro_f1": metrics["macro_f1"], "macro_precision": metrics["macro_precision"],
               "macro_recall": metrics["macro_recall"]}
    summaries.append(summary)
    write_json(output / "frozen_224_384_equal_average.json", metrics)
    print(json.dumps(summary), flush=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(output / "protocol.json", {"checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "split": SPLIT.relative_to(ROOT).as_posix(), "test_images_evaluated": 0,
        "model_weights_updated": False, "validation_images": len(rows),
        "selection_warning": "Resolution candidates examined on the same exploratory validation set"})


if __name__ == "__main__":
    main()
