"""Check focal math, original training-only loader, and one CUDA gradient step."""
import json

import pandas as pd
import torch
from torch.nn import functional as F

from src.data import ROOT, SPLIT, CLASSES, LesionDataset, class_counts, image_paths
from src.losses import MulticlassFocalLoss, class_weights
from src.models import EfficientNetCBAM
from src.train import loader
from src.utils import seed_everything


def main():
    if not SPLIT.is_file():
        raise FileNotFoundError(f"Fixed split is missing: {SPLIT}")
    config = json.loads((ROOT / "configs/focal_v1.json").read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    assignments = pd.read_csv(SPLIT)
    train = assignments.loc[assignments.split == "train"]
    if len(train) != 7009:
        raise ValueError("Original training partition changed")
    paths, duplicates = image_paths()
    if duplicates or any(image_id not in paths for image_id in train.image_id):
        raise ValueError("Training images missing or duplicated")
    alpha = class_weights(class_counts(train))
    print(f"Training images per epoch={len(train)}; no oversampling", flush=True)
    print(f"Alpha weights={dict(zip(CLASSES, alpha.tolist()))}", flush=True)
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    assert torch.allclose(MulticlassFocalLoss(torch.ones(2), gamma=0)(logits, targets), F.cross_entropy(logits, targets))
    assert torch.isfinite(MulticlassFocalLoss(torch.ones(2), gamma=2)(logits, targets))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    batches = loader(LesionDataset(train, paths, training=True, size=config["image_size"]),
                     config["batch_size"], config["workers"], shuffle=True)
    model = EfficientNetCBAM(pretrained=True).to(device).train()
    criterion = MulticlassFocalLoss(alpha, gamma=config["focal_gamma"]).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": config["backbone_lr"]},
        {"params": list(model.cbam.parameters()) + list(model.classifier.parameters()), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    images, labels = next(iter(batches))
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda"):
        loss = criterion(model(images.to(device, non_blocking=True)), labels.to(device, non_blocking=True))
    if not torch.isfinite(loss):
        raise RuntimeError("Focal loss is not finite")
    scaler.scale(loss).backward()
    assert model.cbam[0].mlp[0].weight.grad is not None
    assert model.classifier[-1].weight.grad is not None
    assert model.features[0][0].weight.grad is not None
    scaler.step(optimizer)
    scaler.update()
    print(f"CUDA focal step passed; batch={len(labels)} loss={loss.item():.6f}", flush=True)


if __name__ == "__main__":
    main()
