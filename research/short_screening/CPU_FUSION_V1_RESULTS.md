# CPU-only fusion screening

Saved strict-validation probabilities only. Five lesion-group folds for class-bias and stacking fits; zero meta-train/holdout lesion overlap. Frozen checkpoints previously selected on this same validation cohort, so these are post-test development findings, not independent test estimates. No grid search, model training or test inference.

| Method | Accuracy | Macro-F1 | MEL recall | BKL recall | Gain (pp) | Gained/lost | Positive folds | Material gate |
|---|---:|---:|---:|---:|---:|---|---:|---|
| fixed_equal_probability_reference | 90.1530% | 0.843486 | 0.6084 | 0.8061 | +0.0000 | 0/0 | 0/5 | False |
| fixed_equal_log_probability | 89.4877% | 0.831388 | 0.6145 | 0.8121 | -0.6653 | 10/20 | 1/5 | False |
| crossfit_regularized_class_bias | 89.8204% | 0.837482 | 0.6386 | 0.8121 | -0.3327 | 8/13 | 2/5 | False |
| crossfit_regularized_probability_stacking | 89.0220% | 0.797522 | 0.6145 | 0.7879 | -1.1311 | 13/30 | 0/5 | False |

Oracle: 94.74% if labels were used to choose a correct constituent on every sample. This is not a deployable method or achieved score. 69 reference errors have at least one correct constituent; 79 images are wrong for every member.

Gate was written before outputs: >=1 percentage point accuracy gain, no macro-F1/MEL/BKL recall decline and positive accuracy gain on >=4/5 folds. No extra trials will be added to rescue a failed gate. No final calibrator fitted on the complete evaluation cohort for reporting.
No gate passed: keep the proven reference. These outputs do not justify a GPU run to fit the same fusion or another random backbone.
# Error localization and next decision

Of the 79 validation images missed by all three constituents, 35 are melanoma and18 BKL: 53/79 (67.09%). Those errors cannot be resolved by selecting a correct constituent, because none is correct. A class-score change could change the decision, but must also avoid turning currently correct images into errors; the cross-fitted bias experiment did not improve aggregate accuracy.

The 69 recoverable ensemble errors describe potential diversity, not a working label-free selector. Tested stacking did not realize that potential on held-out lesion folds. Do not report the oracle94.74% as an achieved model result or use it to promise a final score.

No fusion candidate passed the predeclared material gate, so no GPU run is justified by this screen. The next useful research question is training-time improvement in MEL/BKL representation using the proven recipe and a bounded, matched control, rather than additional voting trials. CPU evidence can motivate that hypothesis but cannot establish its training gain. All existing methods, checkpoints and the first/only test result remain frozen evidence.
