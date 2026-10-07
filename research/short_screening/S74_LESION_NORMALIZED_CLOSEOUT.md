# S74: lesion-normalized training influence — rejected

Completed 7 October 2026 in 25.36 seconds on CPU. No GPU, development-validation scoring or original-test inference followed.

## Matched question and result

Does giving each training lesion equal influence before class weights improve generalization over counting every photograph equally? Three fixed lesion-group folds used only the 7,009 development-training images and cached ImageNet ConvNeXt-Tiny features. The unchanged S73 weighted-CE head/control was independently reproduced from its saved checkpoints. Only the candidate's per-image multiplier changed to inverse training-fold lesion view count; architecture, class weights, scaling, seeds, optimizer, schedule and fixed 20-epoch head endpoint stayed matched.

| Train-only grouped out-of-fold metric | Image-weighted control | Lesion-normalized candidate |
|---|---:|---:|
| Accuracy | 79.0840% | 79.1554% |
| Macro precision | 0.617817 | 0.614238 |
| Macro recall | 0.591215 | 0.589256 |
| Macro-F1 | 0.603310 | 0.600311 |
| Melanoma recall | 49.3573% | 48.5861% |
| akiec recall | 48.9083% | 49.7817% |

Candidate gains 101 predictions and loses 96: **net +5 / +0.0713 percentage points**. Fold changes are +9, −3 and −1. It fails the predeclared gate: at least +1 percentage point accuracy, +0.01 macro-F1, nondecreasing melanoma recall and positive net accuracy in at least two folds. No parameter rescue or GPU training is justified. This rejects this specific loss recipe on frozen features; it does not establish that lesion-aware end-to-end training can never help.

Each lesion's summed inverse-view weight equals one within its fitting fold before class weights. The effective loss is normalized by target-weight sum inside each batch; realized SGD gradients need not give every lesion exactly equal influence. Class weights remain based on fitting-fold image counts. No validation counts or labels enter fitting.

## Saved evidence and verification

- Results: `results/short_screening/s74_lesion_normalized_head_cpu/` — plan, log, pooled metrics/predictions/probabilities, fold assignments/weight manifests, class scores, raw/normalized confusion matrices and comparison/training PNG/PDF figures.
- CPU heads/scalers: `checkpoints/short_screening/s74_lesion_normalized_head_cpu/` — three final candidate heads, not CNN checkpoints. These local checkpoint assets are Git-ignored.
- Registry: `results/master_experiment_registry.csv`, row `s74_lesion_normalized_ce_head_train_group_cv_seed42`.
- Runner: `research/short_screening/lesion_normalized_head_screen.py`; rerunning a completed study only reports its saved summary.

Loss invariance/finite backward checks passed. Every image is held out once; fitting/held-out lesions do not overlap. Saved metrics/confusion matrices were recomputed from predictions, and feature/control/checkpoint hashes remained unchanged. Validation feature caches were not loaded; original-test labels/images were not read. The retained **S53 single-image exploratory ensemble remains 93.6128% / 0.886857 macro-F1**; the original frozen test report remains unchanged at **86.7598% / 0.794473**. S74's 79% diagnostic is a different model/cohort and must not be reported as a decline in S53 accuracy.
