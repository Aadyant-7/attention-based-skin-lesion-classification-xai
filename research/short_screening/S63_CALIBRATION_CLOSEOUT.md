# S63: fixed cross-fitted temperature calibration

## Decision

**Reject calibration for the retained method.** It gained only one correct prediction, reduced melanoma recall, and worsened ensemble calibration. Keep S53 as the preferred provisional exploratory candidate. No further temperature/weight search or GPU run was performed.

| Metric | S53 reference | S63 calibration |
|---|---:|---:|
| Exploratory accuracy | 93.6128% | 93.6793% |
| Macro-F1 | 0.886857 | 0.887038 |
| Melanoma recall | 82.0359% | 80.8383% |
| akiec recall | 71.4286% | 71.4286% |
| Correct / incorrect | 1407 / 96 | 1408 / 95 |
| Negative log likelihood | 0.212529 | 0.234706 |
| Ten-bin calibration error | 0.025948 | 0.064936 |

S63 macro precision: **0.917728**; macro recall: **0.864085**. Five predictions became correct and four became incorrect. Fold-wise net changes were +2, +2, 0, -2, -1. BKL and NV gained two correct predictions each; melanoma lost two and BCC lost one.

The predeclared gate required at least eight net additional correct predictions, at least +0.005 absolute accuracy, and nondecreasing macro-F1 and melanoma recall. It failed. The numerical high of 93.6793% is an exploratory study outcome, not an adopted or independently tested model.

## Fixed protocol and verification

- Existing S53 members: S06 ConvNeXt-Tiny, S18 ConvNeXt-Small, S15 DenseNet201, S10 EfficientNetV2-S and S03 EfficientNet-B0; equal 0.20 fusion weights unchanged.
- Five stratified lesion-group meta folds, seed 42; every one of the 1,503 validation images held out exactly once, with zero lesion overlap between fitting and holdout portions.
- One positive temperature per member fitted on each meta-training portion by natural-frequency negative log likelihood. Fixed temperature bounds 0.25–10; no accuracy optimization or weight search. Observed temperatures: 1.558875–2.024841.
- Individual member calibration improved, but the fused ensemble's NLL and calibration error worsened. Individual member argmax predictions did not change.
- Source probabilities/checkpoint hashes verified; original equal fusion reconstructed; output metrics and confusion matrix independently recomputed. Finite normalized probabilities and complete fold coverage verified.
- No raw images, locked-test labels or GPU inference loaded; no training or checkpoint modification. No full-validation temperature fit or deployment was made.

Cross-fitting protects the temperature fitting step only. The underlying models/checkpoints were already selected using this validation cohort; this is **not an independent held-out test estimate**.

## Saved artifacts

`results/short_screening/s63_crossfit_temperature_calibration/` contains the predeclared plan, summary, verification, source manifest, meta-OOF metrics/predictions/probabilities, temperatures, fold assignments/comparisons, member calibration comparisons, class comparison, raw/normalized confusion matrices and PNG/PDF comparison/reliability figures. Registry: `results/master_experiment_registry.csv`, experiment `s63_crossfit_temperature_calibration_exploratory_seed42`.

The original locked-test result remains **86.7598% / 0.794473 macro-F1**. It was not reevaluated or used to fit this calibration study.
