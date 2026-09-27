"""Fast checks including CUDA gradients and a small-sample learning test."""
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import LesionDataset, class_counts, create_or_load_split
from src.losses import balanced_sampler, class_weights
from src.metrics import classification_metrics
from src.models import EfficientNetCBAM


def main():
    rows, paths = create_or_load_split()
    train = rows.loc[rows.split == "train"]
    assert len(rows) == 10015 and len(paths) == 10015
    assert len(class_counts(train)) == 7
    assert torch.isfinite(class_weights(class_counts(train))).all()
    assert len(list(balanced_sampler(train.label.to_numpy()))) == len(train)
    assert classification_metrics([0, 1], [0, 1])["macro_f1"] > 0
    small = train.groupby("label", group_keys=False).head(2)
    dataset = LesionDataset(small, paths, training=False)
    batches = DataLoader(dataset, batch_size=7, shuffle=False, num_workers=0)
    x, y = next(iter(batches))
    assert x.shape == (7, 3, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetCBAM(pretrained=True).to(device)
    assert model(torch.randn(2, 3, 224, 224, device=device)).shape == (2, 7)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x, y = x.to(device), y.to(device)
    model.train()
    started = time.monotonic()
    losses = []
    # Keep one fixed tiny batch, disable dropout randomness, and verify it learns.
    for layer in model.classifier:
        if isinstance(layer, nn.Dropout):
            layer.p = 0.0
    for step in range(30):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            output = model(x)
            loss = criterion(output, y)
        loss.backward()
        if step == 0:
            assert model.cbam[0].mlp[0].weight.grad is not None
            assert model.classifier[-1].weight.grad is not None
            assert model.features[0][0].weight.grad is not None
        optimizer.step()
        losses.append(loss.item())
        if step % 5 == 0:
            print(f"tiny step={step} loss={loss.item():.4f}", flush=True)
    accuracy = (model(x).argmax(1) == y).float().mean().item()
    print(f"tiny initial={losses[0]:.4f} final={losses[-1]:.4f} accuracy={accuracy:.3f} seconds={time.monotonic()-started:.1f}", flush=True)
    if losses[-1] > losses[0] * 0.5 or accuracy < 0.85:
        raise RuntimeError("Tiny-subset overfit failed")
    print("Sanity checks passed", flush=True)


if __name__ == "__main__":
    main()
