# S50–S51 fixed four-view inference closeout

## Result

| Method | Exploratory validation accuracy | Macro-F1 |
|---|---:|---:|
| S46 cached-output equal fusion |93.4797%|.883680|
| S50 fresh FP32 GPU identity control |93.4797%|.883680|
| S51 FP32 four-view TTA |92.2156%|.870358|

Fresh GPU inference independently reproduced ALL cached S46 predicted classes, not just the same rounded score:1405correct/98incorrect,1503images. This confirms the cached fusion arithmetic is reproducible on the real checkpoints. It remains repeated exploratory validation, not independent test validation.

TTA gained8predictions but lost27 versus identity:net−19, −1.2641pp accuracy. Material gate FAILED. Keep identity-only equal fusion. FP32 alone caused no changed decisions; the observed decrease is from the fixed view averaging. Do not search further view subsets or TTA parameters.

## Inputs and artifacts

Exactly S46 ConvNeXt-Tiny/ConvNeXt-Small/DenseNet201/EfficientNetV2-S macro-F1-selected checkpoints,25% each;224RGB bilinear antialias/ImageNet normalization; FP32; view weights25% identity/horizontal/vertical/both. No training, epoch/checkpoint changes, model/weight search, test labels or test images.

Runtime125.7seconds (~2.1minutes) for16model/view passes and closeout. Saved all16source probability CSVs, fused identity/TTA predictions/probabilities, class scores, raw/normalized confusion matrices and publicationPNG/PDF. Registry updated. Paths and exact source hashes recorded in `results/short_screening/s50_s51_current_ensemble_tta/verification.json`.

Verification:1503unique aligned IDs; finite normalized probability vectors; recomputed accuracy/F1/confusion matrices;16complete source passes; identity labels/predictions identical to S46; four original checkpoint hashes unchanged. Original strict architecture and86.7598%one-time test are unchanged. No new test inference occurred.

Decision: retain S46/S50 identity-only93.4797%/.883680 as the strongest observed exploratory candidate. No further TTA variant or GPU training launched.
