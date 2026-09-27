"""Compare normal and horizontal-flip inference on the fixed validation set only."""
import csv
import json
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from src.data import ROOT, SPLIT, LesionDataset, image_paths
from src.metrics import classification_metrics
from src.models import EfficientNetCBAM
from src.train import loader
from src.utils import write_json


RUNS = ("efficientnet_b0_cbam_weighted_v1", "efficientnet_b0_cbam_oversampled_v1")


def evaluate(checkpoint_path, batches, device):
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EfficientNetCBAM(pretrained=False).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    truth, normal, flipped = [], [], []
    normal_nll = tta_nll = 0.0
    with torch.inference_mode():
        for images, labels in batches:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                flip_logits = model(torch.flip(images, dims=(-1,)))
            probabilities = F.softmax(logits.float(), dim=1)
            tta_probabilities = (probabilities + F.softmax(flip_logits.float(), dim=1)) / 2
            normal_nll += F.nll_loss(probabilities.clamp_min(1e-12).log(), labels, reduction="sum").item()
            tta_nll += F.nll_loss(tta_probabilities.clamp_min(1e-12).log(), labels, reduction="sum").item()
            truth.extend(labels.cpu().tolist())
            normal.extend(probabilities.argmax(1).cpu().tolist())
            flipped.extend(tta_probabilities.argmax(1).cpu().tolist())
    return (("normal", normal, normal_nll / len(truth)), ("horizontal_flip_tta", flipped, tta_nll / len(truth))), truth


def main():
    if not SPLIT.is_file():
        raise FileNotFoundError(f"Fixed split is missing: {SPLIT}")
    assignments = pd.read_csv(SPLIT)
    validation = assignments.loc[assignments.split == "val"]
    if len(validation) != 1503 or validation.image_id.duplicated().any():
        raise ValueError("Unexpected validation partition")
    paths, duplicates = image_paths()
    if duplicates or any(image_id not in paths for image_id in validation.image_id):
        raise ValueError("Validation images missing or duplicated")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = loader(LesionDataset(validation, paths, training=False, size=224), 64, 2)
    output = ROOT / "results/validation_inference"
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run_name in RUNS:
        checkpoint = ROOT / "checkpoints" / run_name / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        protocols, truth = evaluate(checkpoint, batches, device)
        for protocol, predicted, nll in protocols:
            metrics = classification_metrics(truth, predicted)
            metrics["validation_unweighted_nll"] = nll
            metrics["run_name"] = run_name
            metrics["protocol"] = protocol
            write_json(output / f"{run_name}_{protocol}.json", metrics)
            summary = {"run_name": run_name, "protocol": protocol, "accuracy": metrics["accuracy"],
                       "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"],
                       "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"],
                       "validation_unweighted_nll": nll}
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    path = ROOT / "results/validation_inference_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved validation-only comparison to {path}", flush=True)


if __name__ == "__main__":
    main()
