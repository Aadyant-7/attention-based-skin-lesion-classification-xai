# Phase 5B image pipeline audit

Only the 7009 saved-split **training** image files were opened for dimension inspection. The black-border sample contains 150 training images selected without replacement using NumPy seed 42. No test image was opened.

## Original dimensions

600×450: 7009

## Implemented transforms

- Training: `Resize((224, 224)) → RandomHorizontalFlip(p=0.5) → ToTensor() → Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
- Validation: `Resize((224, 224)) → ToTensor() → Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
- The tuple resize maps both original axes directly to 224 pixels. It does **not** preserve aspect ratio when the original image is non-square.
- No `CenterCrop`, `RandomResizedCrop`, padding, field-of-view/black-border crop, hair removal, or color-constancy preprocessing is in the CNN training/validation input pipeline.

## Dark-border sample

Near-black means `max(R,G,B) < 30` in the original RGB image. The outer band is 3% of the shorter image side. A sampled image is marked as having an obvious dark border when at least 20% of that band is near-black, **or** at least two of four 5%-side corner patches are at least 50% near-black. This is a simple visual heuristic, not a lesion or artifact detector.

- Obvious dark border: **8/150 (5.3%)**.
- Outer-band near-black fraction: median **0.000**, 90th percentile **0.001**.
- Images with at least 20% near-black outer band: **4/150**.

Per-image sample statistics are in `train_border_sample.csv`.
