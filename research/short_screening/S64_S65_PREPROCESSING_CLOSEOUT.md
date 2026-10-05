# S64/S65: preprocessing screen and consistency check

## Decision

**Do not launch crop-based CNN training from these results.** The full exploratory frozen-feature screen passed, but the predeclared train-only lesion-group consistency check failed. Keep S53's preferred93.6128% exploratory candidate unchanged; S63's93.6793% remains a rejected numerical high, not a deployed improvement.

| Diagnostic | Square224 accuracy / macro-F1 | Official236 + crop224 accuracy / macro-F1 | Decision |
|---|---:|---:|---|
| S64: full exploratory validation, train-only head |79.3081% /0.650621|81.7698% /0.675566|Initial gate passed|
| S65: three train-only lesion-group folds, pooled OOF |75.8025% /0.573473|76.4160% /0.581737|Consistency gate failed|

These are **frozen ImageNet features with newly fitted logistic heads**, not fine-tuned backbone or ensemble performance. Their lower accuracy does not mean the retained ensemble deteriorated. S65 is a training-partition head diagnostic, not a new strict-validation or test score.

## What changed and what was controlled

Same exact torchvision ConvNeXt-Tiny ImageNet1k weights, pooled768-dimensional features, CPU FP32, normalization, complete original cohorts, fixed weighted StandardScaler/logistic regression C1, seed42, no head tuning. Control uses square224 bilinear resize. Candidate uses documented pretrained short-side236 bilinear resize then center crop224. The candidate changes geometry and visible field/cropping jointly. No pretrained weight, existing model checkpoint or historical result was modified.

S64 fitted heads using all7,009 training images and evaluated1,503 exploratory validation images. Both feature sets were freshly computed in CPU FP32; the older FP16 feature control was deliberately not mixed into this comparison. Extraction/reporting took18.1minutes CPU; zero CNN training epochs or GPU activity.

## S64 result detail

-137 formerly incorrect predictions became correct,100 became incorrect: **net+37 /+2.4617percentage points**.
- Macro-F1 rose0.024945; melanoma recall62.8743% ->64.6707%, three additional correct melanoma predictions.
- Per-class net correct changes: akiec+1, BCC-5, BKL+2, DF+1, MEL+3, NV+36, VASC-1. Most of the overall improvement came from nevus, with BCC and vascular recall declining.
- Passed its fixed gate of at least+1percentage point accuracy, no macro-F1 decrease and no melanoma-recall decrease.

## S65 generalization consistency

The fixed three-fold plan was saved before S64 scores were available. Only original training data entered head fitting and assessment. StratifiedGroupKFold grouped by lesion ID, seed42. Every held-out training image was covered exactly once; scaling and sqrt inverse-frequency weights were recomputed within each fitting fold. No validation labels fitted these heads, and no lesion crossed a fold's fitting/holdout boundary.

| Fold | Square accuracy | Crop accuracy | Net additional correct |
|---|---:|---:|---:|
|1|74.8823%|74.5828%|-7|
|2|75.5993%|77.9966%|+56|
|3|76.9264%|76.6695%|-6|

Pooled gain:+43/7,009 images (+0.6135percentage points),612 gained versus569 lost. Only **one of three folds** improved. Melanoma recall50.2571% ->49.8715% (three fewer correct). The fixed gate required+1percentage point, no macro-F1/melanoma-recall decrease, and improvement in at least two folds. It failed. Do not lower the gate or select fold2 alone.

This inconsistency is evidence against treating the S64 result as a reliable general preprocessing fix. It does not prove matched deep fine-tuning could never benefit. These folds still constitute model-development diagnostics, not an untouched final test.

## Verification and artifacts

Both studies independently recomputed saved prediction metrics and confusion matrices and checked complete IDs, finite normalized probabilities and class order. Source weight hashes are recorded; CPU FP32 and no test loading verified. Scores, probability/prediction CSVs, raw/normalized confusion matrices, class scores, comparison PNG/PDF figures and registry entries were saved:

- `results/short_screening/s64_paired_preprocessing_cpu/`
- `results/short_screening/s65_preprocessing_train_group_cv/`
- `.cache/s64_paired_preprocessing_cpu/`: feature arrays and full-training classifier heads.
- `.cache/s65_preprocessing_train_group_cv/`: fold classifier heads.
- `results/master_experiment_registry.csv`: both controls/candidates, clearly separated by validation versus training-meta-OOF evaluation.

There are no new CNN checkpoints or epoch curves, because no CNN training took place. Original locked-test accuracy remains86.7598% /0.794473 macro-F1; no test inference was performed.

## Next substantive action

Move to **external dermoscopy training-data clearance**, not more crop/temperature/weight variants. The metadata-only preflight identified14,885 non-HAM candidates,12,801 with known lesion IDs representing4,255 groups. Notably725 known melanoma groups can be investigated against478 current training groups. Broader relevant cases are a plausible way to improve representation/generalization, not a guaranteed accuracy gain.

First establish label compatibility, published duplicate/alias exclusion, grouping and source policies. No external images were downloaded and no GPU training was launched here. See `EXTERNAL_DATA_NEXT_DIRECTION.md` and `BOTTLENECKS_AND_NEXT_ACTION.md`. A later pilot must have a bounded budget and a prepared command/config before launch. All future test claims need genuinely untouched appropriate data; the original evaluated test is not a tuning resource.
