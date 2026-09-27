"""Cache validation probabilities and frozen post-CBAM embeddings; no test images."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import ROOT, SPLIT, LesionDataset, image_paths
from src.models import EfficientNetCBAM
from src.train import loader


RUNS = ("efficientnet_b0_cbam_weighted_v1", "efficientnet_b0_cbam_oversampled_v1", "efficientnet_b0_cbam_focal_v1")
CACHE = ROOT / ".cache/phase4"


def cache_manifest():
    checkpoints = {name: ROOT / "checkpoints" / name / "best.pt" for name in RUNS}
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return {"split_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
            "checkpoints": {name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for name, path in checkpoints.items()},
            "transform": "deterministic_resize_224_imagenet_normalization", "embedding": "post_cbam_global_pool_1280",
            "tta": "mean_of_original_and_horizontal_flip_probabilities"}


def extract(model, batches, device, embeddings=False):
    vectors, normal, flip = [], [], []
    with torch.inference_mode():
        for images, _ in batches:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                features = model.pool(model.cbam(model.features(images))).flatten(1)
                probabilities = model.classifier(features).float().softmax(1)
                flipped_probabilities = model(torch.flip(images, dims=(-1,))).float().softmax(1)
            if embeddings:
                vectors.append(features.float().cpu().numpy())
            normal.append(probabilities.cpu().numpy())
            flip.append(((probabilities + flipped_probabilities) / 2).cpu().numpy())
    return (np.concatenate(vectors) if embeddings else None,
            np.concatenate(normal), np.concatenate(flip))


def main():
    if not SPLIT.is_file():
        raise FileNotFoundError(f"Fixed split is missing: {SPLIT}")
    manifest = cache_manifest()
    CACHE.mkdir(parents=True, exist_ok=True)
    expected = [CACHE / "weighted_train_embeddings.npz", CACHE / "weighted_val_embeddings.npz"] + [CACHE / f"{name}_val_probabilities.npz" for name in RUNS]
    saved_manifest = CACHE / "manifest.json"
    if saved_manifest.is_file() and json.loads(saved_manifest.read_text(encoding="utf-8")) == manifest and all(path.is_file() for path in expected):
        print("Compatible Phase 4 cache already exists; no CNN forward passes needed", flush=True)
        return
    assignments = pd.read_csv(SPLIT)
    train = assignments.loc[assignments.split == "train"].reset_index(drop=True)
    validation = assignments.loc[assignments.split == "val"].reset_index(drop=True)
    if len(train) != 7009 or len(validation) != 1503 or train.image_id.duplicated().any() or validation.image_id.duplicated().any():
        raise ValueError("Unexpected train/validation partition")
    paths, duplicates = image_paths()
    if duplicates or any(image_id not in paths for image_id in pd.concat([train.image_id, validation.image_id])):
        raise ValueError("Train/validation image mapping is incomplete")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_batches = loader(LesionDataset(train, paths, training=False, size=224), 64, 2)
    val_batches = loader(LesionDataset(validation, paths, training=False, size=224), 64, 2)
    for name in RUNS:
        checkpoint = torch.load(ROOT / "checkpoints" / name / "best.pt", map_location=device, weights_only=False)
        model = EfficientNetCBAM(pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        if name == RUNS[0]:
            train_vectors = extract_train_embeddings(model, train_batches, device)
            if train_vectors.shape != (7009, 1280):
                raise ValueError(f"Unexpected train embedding shape: {train_vectors.shape}")
            np.savez_compressed(CACHE / "weighted_train_embeddings.npz", ids=train.image_id.to_numpy(dtype=str), y=train.label.to_numpy(), X=train_vectors)
        val_vectors, normal, tta = extract(model, val_batches, device, embeddings=name == RUNS[0])
        if normal.shape != (1503, 7) or tta.shape != (1503, 7):
            raise ValueError("Unexpected validation probability shape")
        np.savez_compressed(CACHE / f"{name}_val_probabilities.npz", ids=validation.image_id.to_numpy(dtype=str), y=validation.label.to_numpy(), normal=normal, tta=tta)
        if val_vectors is not None:
            if val_vectors.shape != (1503, 1280):
                raise ValueError(f"Unexpected validation embedding shape: {val_vectors.shape}")
            np.savez_compressed(CACHE / "weighted_val_embeddings.npz", ids=validation.image_id.to_numpy(dtype=str), y=validation.label.to_numpy(), X=val_vectors)
        print(f"Cached {name}: validation normal/TTA probabilities" + (" and frozen embeddings" if val_vectors is not None else ""), flush=True)
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    saved_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Phase 4 cache complete; test images were not loaded", flush=True)


def extract_train_embeddings(model, batches, device):
    vectors = []
    with torch.inference_mode():
        for images, _ in batches:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                features = model.pool(model.cbam(model.features(images))).flatten(1)
            vectors.append(features.float().cpu().numpy())
    return np.concatenate(vectors)


if __name__ == "__main__":
    main()
