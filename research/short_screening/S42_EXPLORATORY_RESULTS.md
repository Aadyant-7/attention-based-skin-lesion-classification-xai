# S42 exploratory CPU fusion closeout

## Result

**93.1470% exploratory validation accuracy / .879744 macro-F1.**1400correct/103incorrect on the existing1503-image image-level validation cohort. No GPU, training or test inference.

Members: S06 ConvNeXt-Tiny + S18 ConvNeXt-Small + S15 DenseNet201 + S10 EfficientNetV2-S, original accuracy-selected checkpoints. Equal probability weights .25 each. All source scores/probabilities preserved; no new weights or checkpoint selection. Source inference precision is inherited, not newly claimed uniformly FP32.

| Method | Exploratory accuracy | Macro-F1 |
|---|---:|---:|
| Frozen S13 reference |92.4817%|.877454|
| Historical S27 numerical best |92.7478%|.866355|
| S42 fixed equal four |93.1470%|.879744|
| S42 lesion-group cross-fitted learned stack |91.7498%|.854805|

The previous user-facing92.61% overall-best claim was incomplete: saved S27 already reached92.7478%. That historical result remains unchanged. S42 now exceeds that numerical best by .3992pp, gaining26 and losing20 predictions (net6). Against the frozen reference the aggregate gain is .6653pp (net10).

The predeclared material gate against S27 required +.5pp, F1>=.877454 and positive gain on>=4/5 diagnostic folds. S42 does NOT pass: gain+.3992pp; only2/5 fold gains positive. Report the actual new high, but do not claim a substantial/statistically established improvement or93% test accuracy. No search was added after seeing results.

The learned fusion failed. Do not keep adjusting C/weights/folds to rescue it. Fold heldout lesions had zero overlap with meta-training lesions, but source CNNs were already selected on the full validation cohort; original exploratory training can share lesions with validation. These scores remain developmental, not independent CNN cross-validation.

## Verified artifacts

Prediction IDs/classes match the manifest;1503finite normalized distributions; exact equal fusion recomputed; accuracy, macro-F1 and matrix independently recomputed. Source prediction hashes remain unchanged. Checkpoint paths/hashes and pretrained initialization recorded in `candidate_manifest.json`. Metrics, per-class scores, raw/normalized matrices, complete prediction/probability CSVs and PNG/PDF figures saved. Registry updated.

Original strict frozen architecture,90.1530% validation and86.7598% one-time test remain separate and unchanged. Original test was not accessed. Do not replace those test results with S42.

## Decision

Preserve S42 as the strongest observed exploratory candidate, with both the modest gain and evaluation limitations disclosed. No additional GPU run is justified merely to reproduce the already-computed fusion. No new model, large search, or test rerun launched.
