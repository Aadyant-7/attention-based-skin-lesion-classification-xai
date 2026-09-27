# Phase 5C validation-only prior and cost correction

Target prior was computed exclusively from the original 7,009 **training** split rows. Oversampled source prior was exactly uniform. Weighted-CE correction is an exploratory cost-weight adjustment, not exact Bayesian calibration.

For both normal and horizontal-flip-TTA probability streams, the saved grid tests lambda = 0.0, 0.1, ..., 1.0. Corrected probabilities are normalized after adding the log shift. Full per-class F1 and confusion matrices for all 44 grid candidates are in `correction_grid.csv`.

Best corrected accuracy: `oversampled_tta_lambda_0.2` = 0.858949 accuracy, 0.722723 macro F1.
Best corrected macro F1: `weighted_tta_lambda_0.1` = 0.843646 accuracy, 0.766175 macro F1.
Existing image-only references: 0.857618 best accuracy; 0.771810 best macro F1.

Best corrected accuracy versus its own uncorrected stream: NV F1 0.931 → 0.930; MEL F1 0.606 → 0.606; mel↔nv errors 80 → 79.
Best corrected macro F1 versus its own uncorrected stream: NV F1 0.921 → 0.921; MEL F1 0.544 → 0.544; mel↔nv errors 103 → 103.

The best corrected-accuracy stream changes predicted NV count from 1051 to 1080; correct NV predictions 958 → 970, correct MEL predictions 90 → 86. MEL→NV rises 54 → 62, while NV→MEL falls 26 → 17. The small accuracy gain is driven by majority-class predictions, not balanced class improvement.
Optional three-weight combination with the existing Phase 4 ensemble did not help: its best macro F1 was 0.766421 (accuracy 0.846307).

The full correction is a theoretical diagnostic. Coarse lambda selection uses validation labels and is exploratory; it is not a held-out estimate. No model was trained and no test data were used.
