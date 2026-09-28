"""Four-epoch strict-development pilot: PanDerm Base, last two blocks and head.

The locked test images are never opened. Stop after epoch 2 if validation
accuracy has not cleared 85.5%; preserve latest and best checkpoints.
"""
import csv
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import ROOT, SPLIT, image_paths
from src.metrics import classification_metrics
from src.utils import atomic_save, seed_everything, write_json
from scripts.probe_panderm_base import Images, TRANSFORM, load_backbone


RUN = "panderm_base_last2_pilot_v1"
OUTPUT = ROOT / "results/accuracy_exploration" / RUN
CHECKPOINT = ROOT / "checkpoints/accuracy_exploration" / RUN
HEAD = ROOT / "checkpoints/accuracy_exploration/panderm_base_strict_heads_v1/standardized_logreg_c1.joblib"
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.228, .224, .225)),
])


def initialize_head(model):
    pipeline = joblib.load(HEAD)
    scaler, classifier = pipeline[0], pipeline[1]
    weights = classifier.coef_ / scaler.scale_[None, :]
    bias = classifier.intercept_ - weights @ scaler.mean_
    model.head = nn.Linear(768, 7)
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(weights).float())
        model.head.bias.copy_(torch.from_numpy(bias).float())


def evaluate(model, batches, device):
    model.eval()
    y_true, probs = [], []
    with torch.inference_mode():
        for pixels, labels in batches:
            with torch.autocast(device_type="cuda"):
                logits = model(pixels.to(device, non_blocking=True))
            probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
            y_true.extend(labels.tolist())
    probabilities = np.concatenate(probs)
    return classification_metrics(y_true, probabilities.argmax(axis=1)), probabilities


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot requires CUDA")
    seed_everything(42)
    frame = pd.read_csv(SPLIT)
    train = frame.loc[frame.split == "train"].reset_index(drop=True)
    val = frame.loc[frame.split == "val"].reset_index(drop=True)
    test = frame.loc[frame.split == "test"]
    if len(train) != 7009 or len(val) != 1503 or len(test) != 1503 or set(train.lesion_id) & set(val.lesion_id):
        raise ValueError("Strict development split changed")
    if set(train.lesion_id) & set(test.lesion_id) or set(val.lesion_id) & set(test.lesion_id):
        raise ValueError("Strict test lesion overlap")
    paths, duplicates = image_paths()
    if duplicates or any(image_id not in paths for image_id in pd.concat([train, val]).image_id):
        raise ValueError("Missing or duplicate development images")
    train_batches = DataLoader(Images(train, paths, TRAIN_TRANSFORM), batch_size=32, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_batches = DataLoader(Images(val, paths, TRANSFORM), batch_size=32, shuffle=False,
                             num_workers=2, pin_memory=True)
    device = torch.device("cuda")
    model = load_backbone(device)
    initialize_head(model)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in model.blocks[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for layer in (model.norm, model.head):
        for parameter in layer.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": model.blocks[-2:].parameters(), "lr": 5e-6},
        {"params": model.norm.parameters(), "lr": 5e-6},
        {"params": model.head.parameters(), "lr": 5e-5},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    best_accuracy, best_epoch = -1.0, 0
    history = []
    started = time.time()
    for epoch in range(1, 5):
        model.eval()
        for layer in model.blocks[-2:]:
            layer.train()
        model.head.train()
        total_loss = 0.0
        for pixels, labels in train_batches:
            optimizer.zero_grad(set_to_none=True)
            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda"):
                logits = model(pixels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(labels)
        metrics, probabilities = evaluate(model, val_batches, device)
        row = {"epoch": epoch, "train_loss": total_loss / len(train),
               "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
               "elapsed_seconds": time.time() - started}
        history.append(row)
        payload = {"run": RUN, "epoch": epoch, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                   "validation": metrics}
        atomic_save(payload, CHECKPOINT / "latest.pt")
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = metrics["accuracy"], epoch
            atomic_save(payload, CHECKPOINT / "best.pt")
            np.savez_compressed(ROOT / ".cache/panderm_base_strict_lp_v1/pilot_best_val_probabilities.npz",
                                ids=val.image_id.to_numpy(dtype=str), y=val.label.to_numpy(),
                                probabilities=probabilities)
            write_json(OUTPUT / "best_validation_metrics.json", metrics)
        with (OUTPUT / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps({**row, "best_epoch": best_epoch, "best_accuracy": best_accuracy}), flush=True)
        if epoch == 2 and best_accuracy <= .855:
            print("Pilot stop: no convincing early accuracy gain", flush=True)
            break
    write_json(OUTPUT / "protocol.json", {"run": RUN, "train_images": len(train),
        "validation_images": len(val), "test_images_evaluated": 0,
        "backbone": "PanDerm Base; last two transformer blocks plus layer norm and head updated",
        "initial_head": "standardized logistic regression C=1 on strict training embeddings",
        "max_epochs": 4, "completed_epochs": len(history), "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy, "checkpoint": (CHECKPOINT / "best.pt").relative_to(ROOT).as_posix(),
        "selection_warning": "Best checkpoint selected on strict validation accuracy"})


if __name__ == "__main__":
    main()
