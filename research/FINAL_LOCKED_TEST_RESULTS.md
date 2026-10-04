# Final locked-test results â€” first and only held-out evaluation

This is the first and only locked-test evaluation in the documented structured workflow. Test labels/images played no role in architecture selection, hyperparameter selection, ensemble weighting or checkpoint selection. Predictions from each frozen constituent were generated exactly once, with labels withheld from inference; standalone scores reuse those same probabilities. No alternative test method was evaluated and no methodology was modified after the result.

Frozen S31: S28 B0 epoch19 + S29 ConvNeXt-Tiny epoch33 + S30 EfficientNetV2-S epoch24. Equal probability averaging (1/3 each), FP32, identity only,224x224 bilinear antialias,RGB/ImageNet normalization. No inference augmentation,TTA,multi-resolution or tuned weights. Exact paths/hashes, unchanged config/source and zero lesion overlap verification are recorded in `results/final_locked_test/v1/pre_inference_verification.json`.

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| efficientnet_b0 | 82.1690% | 0.691421 | 0.698191 | 0.694072 | 0.820253 |
| convnext_tiny | 85.0299% | 0.766967 | 0.734878 | 0.749438 | 0.846641 |
| efficientnet_v2_s | 83.9654% | 0.751929 | 0.726672 | 0.737212 | 0.833923 |
| S31 equal ensemble | 86.7598% | 0.814010 | 0.778724 | 0.794473 | 0.863082 |

## S31 class performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| akiec | 0.822222 | 0.755102 | 0.787234 | 49 |
| bcc | 0.768293 | 0.807692 | 0.787500 | 78 |
| bkl | 0.793333 | 0.721212 | 0.755556 | 165 |
| df | 0.705882 | 0.705882 | 0.705882 | 17 |
| mel | 0.699248 | 0.553571 | 0.617940 | 168 |
| nv | 0.909091 | 0.955224 | 0.931587 | 1005 |
| vasc | 1.000000 | 0.952381 | 0.975610 | 21 |

Correct 1304; incorrect 199; cohort1503. Melanoma recall 0.553571; akiec recall 0.755102.

## Confusion matrix

Rows=true,columns=predicted; order ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'].

```text
[[ 37   3   3   2   1   3   0]
 [  2  63   2   0   1  10   0]
 [  2   5 119   1  14  24   0]
 [  2   1   0  12   1   1   0]
 [  1   2  15   0  93  57   0]
 [  1   8  11   2  23 960   0]
 [  0   0   0   0   0   1  20]]
```

## Descriptive comparison

Exploratory validation 92.4817% / 0.877454; strict validation 90.1530% / 0.843486; locked test 86.7598% / 0.794473.

Test minus strict: -3.3932 accuracy percentage points; -0.049014 macro-F1. Test minus exploratory: -5.7219 percentage points; -0.082981 macro-F1. Different cohorts/protocols are not interchangeable. Single split evidence does not establish robustness or statistical superiority; interpret melanoma/akiec recall separately from majority-class accuracy.

Outputs: `results/final_locked_test/v1/`: full probabilities/predictions, raw and normalized matrices, per-class scores, and publication PNG/PDF figures. No training checkpoints were overwritten.

Strict validation was optimistic by 3.3932 accuracy percentage points and 0.049014 macro-F1. The method retained useful held-out performance and the ensemble exceeded all constituents, but the estimate did not transfer closely; melanoma sensitivity is a clear limitation (55.36% test versus 60.84% strict validation). This is descriptive evidence from one held-out cohort, not proof of robustness.

Performance work is FINISHED. No additional test inference, architecture selection or optimization follows this report. Next phase: Grad-CAM/XAI, final figures/tables and paper writing.
