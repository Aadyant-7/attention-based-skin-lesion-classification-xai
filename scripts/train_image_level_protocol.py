"""Exploratory image-level protocol probe; never uses the strict test images.

The architecture, input transform, loss, and optimization match first_run.json.
Only the train/validation assignment within the strict development pool changes.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import transforms

from src.data import ROOT, SPLIT, MEAN, STD, LesionDataset, class_counts, image_paths
from src.losses import class_weights
from src.models import EfficientNetCBAM
from src.train import epoch_pass, loader
from src.utils import atomic_save, seed_everything, write_json


RUN_NAME = "image_level_weighted_b0_cbam_v1"
STRONG_RUN_NAME = "image_level_strong_aug_b0_cbam_v1"
SPLIT_PATH = ROOT / "data/splits/exploratory/image_level_dev_v1.csv"


def training_dataset(rows, paths, recipe):
    dataset = LesionDataset(rows, paths, training=True, size=224)
    if recipe == "strong_aug":
        dataset.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=.5),
            transforms.RandomVerticalFlip(p=.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(.05, .05), scale=(.95, 1.05)),
            transforms.ColorJitter(brightness=.12, contrast=.12, saturation=.10, hue=.02),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return dataset


def make_or_check_split():
    strict = pd.read_csv(SPLIT)
    assert len(strict) == 10015 and strict.image_id.is_unique
    strict_test = strict.loc[strict.split == "test"].copy()
    dev = strict.loc[strict.split != "test"].copy()
    assert len(dev) == 8512 and len(strict_test) == 1503
    train_ids, val_ids = train_test_split(
        dev.image_id.to_numpy(), test_size=1503, stratify=dev.diagnosis.to_numpy(), random_state=42
    )
    proposed = strict.copy()
    proposed.loc[proposed.image_id.isin(train_ids), "split"] = "train"
    proposed.loc[proposed.image_id.isin(val_ids), "split"] = "val"
    if SPLIT_PATH.exists():
        saved = pd.read_csv(SPLIT_PATH)
        if not saved.equals(proposed):
            raise ValueError(f"Exploratory split changed: {SPLIT_PATH}")
    else:
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(SPLIT_PATH, index=False)
    if not proposed.loc[proposed.image_id.isin(strict_test.image_id)].equals(strict_test):
        raise AssertionError("Strict test assignment changed")
    if set(proposed.loc[proposed.split != "test", "image_id"]) & set(strict_test.image_id):
        raise AssertionError("Strict test image entered development pool")
    train = proposed.loc[proposed.split == "train"]
    val = proposed.loc[proposed.split == "val"]
    assert len(train) == 7009 and len(val) == 1503
    assert set(train.image_id).isdisjoint(val.image_id)
    shared_lesions = set(train.lesion_id) & set(val.lesion_id)
    diagnostics = {
        "protocol": "EXPLORATORY image-level published-protocol comparison; not lesion-disjoint",
        "strict_test_excluded": True,
        "strict_test_images": len(strict_test),
        "train_images": len(train),
        "validation_images": len(val),
        "train_validation_shared_lesions": len(shared_lesions),
        "validation_images_with_train_lesion": int(val.lesion_id.isin(shared_lesions).sum()),
        "train_class_counts": train.diagnosis.value_counts().sort_index().to_dict(),
        "validation_class_counts": val.diagnosis.value_counts().sort_index().to_dict(),
    }
    return train, val, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--recipe", choices=("baseline", "strong_aug"), default="baseline")
    args = parser.parse_args()
    run_name = RUN_NAME if args.recipe == "baseline" else STRONG_RUN_NAME
    run_dir = ROOT / "results/exploratory/runs" / run_name
    checkpoint_dir = ROOT / "checkpoints/exploratory" / run_name
    config = json.loads((ROOT / "configs/first_run.json").read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    train, val, diagnostics = make_or_check_split()
    print(json.dumps(diagnostics, indent=2), flush=True)
    if args.preflight:
        paths, duplicates = image_paths()
        if duplicates or len(paths) < 10015:
            raise ValueError("Missing or duplicate image paths")
        sample = training_dataset(train.head(2), paths, args.recipe)
        x, _ = sample[0]
        assert tuple(x.shape) == (3, 224, 224)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        model = EfficientNetCBAM(pretrained=True).cuda().eval()
        with torch.inference_mode():
            out = model(x.unsqueeze(0).cuda())
        assert tuple(out.shape) == (1, 7) and torch.isfinite(out).all()
        print("PREFLIGHT_OK CUDA=" + torch.cuda.get_device_name(0), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if (checkpoint_dir / "latest.pt").exists():
        raise FileExistsError("This probe has already started; refusing to overwrite checkpoints")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    paths, duplicates = image_paths()
    if duplicates or len(paths) < 10015:
        raise ValueError("Missing or duplicate image paths")
    train_batches = loader(training_dataset(train, paths, args.recipe), 64, 2, shuffle=True)
    val_batches = loader(LesionDataset(val, paths, size=224), 64, 2)
    model = EfficientNetCBAM(pretrained=True).to(device)
    weights = class_weights(class_counts(train)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": config["backbone_lr"]},
        {"params": list(model.cbam.parameters()) + list(model.classifier.parameters()), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    scaler = torch.amp.GradScaler("cuda")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {**config, "run_name": run_name, "recipe": args.recipe, "protocol": diagnostics})
    history = run_dir / "history.csv"
    best_f1, best_epoch, best_accuracy, stale = -1., 0, 0., 0
    started = time.monotonic()
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = epoch_pass(model, train_batches, criterion, device, optimizer, scaler)
        val_metrics = epoch_pass(model, val_batches, criterion, device)
        scheduler.step(val_metrics["macro_f1"])
        best_accuracy = max(best_accuracy, val_metrics["accuracy"])
        improved = val_metrics["macro_f1"] > best_f1 + 1e-5
        if improved:
            best_f1, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
        else:
            stale += 1
        record = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
                  "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
                  "val_macro_precision": val_metrics["macro_precision"], "val_macro_recall": val_metrics["macro_recall"],
                  "val_macro_f1": val_metrics["macro_f1"]}
        with history.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(record))
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(record)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "scaler": scaler.state_dict(), "epoch": epoch, "best_epoch": best_epoch, "best_f1": best_f1,
                   "config": config, "protocol": diagnostics}
        atomic_save(payload, checkpoint_dir / "latest.pt")
        if improved:
            atomic_save(payload, checkpoint_dir / "best.pt")
            write_json(run_dir / "validation_metrics.json", val_metrics)
        print(f"epoch={epoch} train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} best_f1={best_f1:.4f} stale={stale}", flush=True)
        if stale >= config["patience"]:
            print("Early stopping", flush=True)
            break
        # A poor trajectory at epoch 12 rules out this split as a useful high-accuracy reproduction.
        if epoch >= 12 and best_accuracy < .82:
            print("Kill criterion: best accuracy below 0.82 after 12 epochs", flush=True)
            break
    best = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    write_json(ROOT / "results/exploratory" / (run_name + "_summary.json"), {
        "run_name": run_name, "recipe": args.recipe, "protocol": diagnostics, "epochs": epoch, "best_epoch": best_epoch,
        "best_validation": best, "runtime_seconds": round(time.monotonic()-started, 1),
        "checkpoint": (checkpoint_dir / "best.pt").relative_to(ROOT).as_posix(),
        "status": "completed", "strict_test_evaluated": False,
    })
    print("Training complete", flush=True)


if __name__ == "__main__":
    main()
