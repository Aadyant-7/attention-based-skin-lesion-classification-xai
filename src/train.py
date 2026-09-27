"""Validation-selected training; the test partition is never loaded here."""
import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ROOT, LesionDataset, class_counts, create_or_load_split
from .losses import balanced_sampler, class_weights
from .metrics import classification_metrics
from .models import EfficientNetCBAM
from .utils import atomic_save, restore_rng, rng_state, seed_everything, write_json


def loader(dataset, batch_size, workers, shuffle=False, sampler=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0)


def epoch_pass(model, batches, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels, predictions = [], []
    for x, y in batches:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * len(y)
        labels.extend(y.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = total_loss / len(labels)
    return metrics


def upsert_experiment(row):
    run_name = row.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("Experiment row requires a nonempty run_name")
    path = ROOT / "results/experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            existing = list(csv.DictReader(file))
    if any(None in item for item in existing):
        raise ValueError(f"Malformed experiment CSV with extra columns: {path}")
    existing = [item for item in existing if item.get("run_name") != run_name]
    existing.append(row)
    fields = list(dict.fromkeys(key for item in existing for key in item))
    temp = path.with_suffix(".csv.tmp")
    try:
        with temp.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(existing)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/first_run.json")
    parser.add_argument("--resume", action="store_true", help="Require and restore latest checkpoint")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("Serious training requires CUDA")
    device = torch.device("cuda")
    rows, paths = create_or_load_split()
    train_rows = rows.loc[rows.split == "train"]
    val_rows = rows.loc[rows.split == "val"]
    counts = class_counts(train_rows)
    strategy = config["imbalance"]
    if strategy not in {"class_weighted", "balanced_sampler"}:
        raise ValueError(strategy)
    sampler = balanced_sampler(train_rows.label.to_numpy(), config["seed"]) if strategy == "balanced_sampler" else None
    criterion = nn.CrossEntropyLoss(weight=class_weights(counts).to(device) if strategy == "class_weighted" else None)
    train_batches = loader(LesionDataset(train_rows, paths, training=True, size=config["image_size"]), config["batch_size"], config["workers"], shuffle=sampler is None, sampler=sampler)
    val_batches = loader(LesionDataset(val_rows, paths, size=config["image_size"]), config["batch_size"], config["workers"])
    model = EfficientNetCBAM(pretrained=True).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": config["backbone_lr"]},
        {"params": list(model.cbam.parameters()) + list(model.classifier.parameters()), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda")
    run = ROOT / "results/runs" / config["run_name"]
    checkpoint_dir = ROOT / "checkpoints" / config["run_name"]
    run.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(run / "config.json", config)
    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best.pt"
    start, best_f1, best_epoch, stale = 1, -1.0, 0, 0
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["config"] != config:
            raise ValueError("Existing checkpoint config differs; use a new run name")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        start, best_f1, best_epoch, stale = saved["epoch"] + 1, saved["best_f1"], saved["best_epoch"], saved["stale"]
        print(f"Checkpoint found. Saved epoch: {saved['epoch']}. Best validation metric: {best_f1:.4f}. Resuming from epoch: {start}. Next epoch: {start}", flush=True)
    elif args.resume:
        raise FileNotFoundError(latest)
    print(f"CUDA: {torch.cuda.get_device_name(0)} | batch={config['batch_size']} | train={len(train_rows)} val={len(val_rows)} | strategy={strategy} | weights={class_weights(counts).tolist() if strategy == 'class_weighted' else 'none'}", flush=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "run_name": config["run_name"],
        "variant": "EfficientNet-B0", "image_size": config["image_size"], "pretrained": True, "cbam": "reduction=16,kernel=7",
        "classifier_dimensions": "1280-512-128-7", "dropout": "0.3-0.2-0.1", "imbalance": strategy,
        "loss": "weighted CE" if strategy == "class_weighted" else "CE", "weight_formula": "sqrt(N/(K*n_c))/mean" if strategy == "class_weighted" else "none",
        "augmentation": "horizontal flip p=0.5", "optimizer": "AdamW", "backbone_lr": config["backbone_lr"],
        "head_lr": config["head_lr"], "weight_decay": config["weight_decay"], "batch_size": config["batch_size"],
        "epochs": config["epochs"], "best_epoch": "", "val_accuracy": "", "val_macro_precision": "", "val_macro_recall": "",
        "val_macro_f1": "", "val_loss": "", "runtime_seconds": "", "checkpoint": best.relative_to(ROOT).as_posix(), "status": "running",
        "notes": "Validation selected; test untouched"}
    upsert_experiment(record)
    started = time.monotonic()
    history = run / "history.csv"
    for epoch in range(start, config["epochs"] + 1):
        train_metrics = epoch_pass(model, train_batches, criterion, device, optimizer, scaler)
        val_metrics = epoch_pass(model, val_batches, criterion, device)
        scheduler.step(val_metrics["macro_f1"])
        score = val_metrics["macro_f1"]
        improved = score > best_f1 + 1e-5
        if improved:
            best_f1, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        history_record = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
                  "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
                  "val_macro_precision": val_metrics["macro_precision"], "val_macro_recall": val_metrics["macro_recall"], "val_macro_f1": score}
        with history.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history_record))
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(history_record)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "scaler": scaler.state_dict(), "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch,
                   "stale": stale, "config": config, "rng": rng_state()}
        atomic_save(payload, latest)
        if improved:
            atomic_save(payload, best)
            write_json(run / "validation_metrics.json", val_metrics)
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={score:.4f} best={best_f1:.4f} stale={stale}", flush=True)
        if stale >= config["patience"]:
            print("Early stopping", flush=True)
            break
    best_metrics = json.loads((run / "validation_metrics.json").read_text(encoding="utf-8"))
    record.update({"best_epoch": best_epoch, "val_accuracy": best_metrics["accuracy"],
        "val_macro_precision": best_metrics["macro_precision"], "val_macro_recall": best_metrics["macro_recall"],
        "val_macro_f1": best_metrics["macro_f1"], "val_loss": best_metrics["loss"],
        "runtime_seconds": round(time.monotonic() - started, 1), "checkpoint": best.relative_to(ROOT).as_posix(), "status": "completed",
        "notes": "Validation selected; test untouched"})
    upsert_experiment(record)
    print("Training complete", flush=True)


if __name__ == "__main__":
    main()
