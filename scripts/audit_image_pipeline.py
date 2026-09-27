"""Read-only audit of training images and the implemented input transforms."""
import csv
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from src.data import MEAN, STD, ROOT, SPLIT, image_paths, transform


OUT = ROOT / "results/phase5b_inference_audit"


def dark_border_stats(path):
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("RGB"))
    height, width = pixels.shape[:2]
    band = max(1, round(0.03 * min(height, width)))
    near_black = pixels.max(axis=2) < 30
    edge = np.zeros((height, width), dtype=bool)
    edge[:band, :] = edge[-band:, :] = True
    edge[:, :band] = edge[:, -band:] = True
    outer_fraction = float(near_black[edge].mean())
    corner_size = max(1, round(0.05 * min(height, width)))
    corners = (near_black[:corner_size, :corner_size], near_black[:corner_size, -corner_size:],
               near_black[-corner_size:, :corner_size], near_black[-corner_size:, -corner_size:])
    dark_corners = sum(float(corner.mean()) >= 0.5 for corner in corners)
    obvious = outer_fraction >= 0.2 or dark_corners >= 2
    return outer_fraction, dark_corners, obvious


def main():
    if not SPLIT.is_file():
        raise FileNotFoundError(f"Saved split is missing: {SPLIT}")
    rows = pd.read_csv(SPLIT)
    train = rows.loc[rows.split == "train"]
    if len(train) != 7009:
        raise ValueError("Unexpected training partition")
    paths, duplicates = image_paths()
    if duplicates or any(image_id not in paths for image_id in train.image_id):
        raise ValueError("Training image mapping incomplete")
    dimensions = Counter()
    for image_id in train.image_id:
        with Image.open(paths[image_id]) as image:
            dimensions[image.size] += 1
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(train.image_id.to_numpy(dtype=str), size=150, replace=False)
    sample = []
    for image_id in sample_ids:
        fraction, corners, obvious = dark_border_stats(paths[image_id])
        sample.append({"image_id": image_id, "outer_near_black_fraction": fraction,
                       "near_black_corners": corners, "obvious_dark_border": int(obvious)})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "train_border_sample.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)
    fractions = np.array([row["outer_near_black_fraction"] for row in sample])
    obvious_count = sum(row["obvious_dark_border"] for row in sample)
    dimensions_text = ", ".join(f"{width}×{height}: {count}" for (width, height), count in sorted(dimensions.items()))
    text = f"""# Phase 5B image pipeline audit

Only the {len(train)} saved-split **training** image files were opened for dimension inspection. The black-border sample contains 150 training images selected without replacement using NumPy seed 42. No test image was opened.

## Original dimensions

{dimensions_text}

## Implemented transforms

- Training: `Resize((224, 224)) → RandomHorizontalFlip(p=0.5) → ToTensor() → Normalize(mean={list(MEAN)}, std={list(STD)})`.
- Validation: `Resize((224, 224)) → ToTensor() → Normalize(mean={list(MEAN)}, std={list(STD)})`.
- The tuple resize maps both original axes directly to 224 pixels. It does **not** preserve aspect ratio when the original image is non-square.
- No `CenterCrop`, `RandomResizedCrop`, padding, field-of-view/black-border crop, hair removal, or color-constancy preprocessing is in the CNN training/validation input pipeline.

## Dark-border sample

Near-black means `max(R,G,B) < 30` in the original RGB image. The outer band is 3% of the shorter image side. A sampled image is marked as having an obvious dark border when at least 20% of that band is near-black, **or** at least two of four 5%-side corner patches are at least 50% near-black. This is a simple visual heuristic, not a lesion or artifact detector.

- Obvious dark border: **{obvious_count}/{len(sample)} ({obvious_count/len(sample):.1%})**.
- Outer-band near-black fraction: median **{np.median(fractions):.3f}**, 90th percentile **{np.quantile(fractions, 0.9):.3f}**.
- Images with at least 20% near-black outer band: **{int((fractions >= 0.2).sum())}/{len(sample)}**.

Per-image sample statistics are in `train_border_sample.csv`.
"""
    (OUT / "preprocessing_audit.md").write_text(text, encoding="utf-8")
    print(dimensions_text, flush=True)
    print(f"Obvious dark borders: {obvious_count}/150; outer near-black median={np.median(fractions):.3f}", flush=True)


if __name__ == "__main__":
    main()
