"""Verify fixed-split oversampling and a few real CUDA training batches."""
import json
import math
import time

import torch
from torch import nn

from src.data import ROOT, SPLIT, CLASSES, LesionDataset, class_counts, create_or_load_split, oversample_training_rows
from src.models import EfficientNetCBAM
from src.train import loader
from src.utils import seed_everything


def main():
    if not SPLIT.is_file():
        raise FileNotFoundError(f"Fixed split is missing: {SPLIT}")
    config = json.loads((ROOT / "configs/oversampled_v1.json").read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    rows, paths = create_or_load_split()
    train = rows.loc[rows.split == "train"]
    validation = rows.loc[rows.split == "val"]
    test = rows.loc[rows.split == "test"]
    sampled = oversample_training_rows(train, config["seed"])
    assert set(sampled.image_id).issubset(set(train.image_id))
    assert not (set(sampled.lesion_id) & set(validation.lesion_id))
    assert not (set(sampled.lesion_id) & set(test.lesion_id))
    assert len(validation) == 1503 and len(test) == 1503
    print(f"Original training class counts: {dict(zip(CLASSES, class_counts(train).tolist()))}", flush=True)
    print(f"Effective training class counts: {dict(zip(CLASSES, class_counts(sampled).tolist()))}", flush=True)
    print(f"Effective samples per epoch: {len(sampled)}", flush=True)
    print(f"Validation rows={len(validation)} test rows={len(test)}; neither was sampled", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    batches = loader(LesionDataset(sampled, paths, training=True, size=config["image_size"]),
                     config["batch_size"], config["workers"], shuffle=True)
    model = EfficientNetCBAM(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    assert criterion.weight is None
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": config["backbone_lr"]},
        {"params": list(model.cbam.parameters()) + list(model.classifier.parameters()), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    seconds = []
    for number, (x, y) in enumerate(batches):
        if number == 4:
            break
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            loss = criterion(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        seconds.append(time.monotonic() - start)
        print(f"CUDA batch {number + 1}: loss={loss.item():.4f} seconds={seconds[-1]:.2f}", flush=True)
    steady = sum(seconds[1:]) / len(seconds[1:])
    train_batches = math.ceil(len(sampled) / config["batch_size"])
    val_batches = math.ceil(len(validation) / config["batch_size"])
    print(f"CUDA device={torch.cuda.get_device_name(0)}; estimated train compute per epoch={train_batches * steady / 60:.1f} min plus validation/data/checkpoint overhead; batches={train_batches}+{val_batches}", flush=True)
    print("Preflight passed; no checkpoint or experiment row was written", flush=True)


if __name__ == "__main__":
    main()
