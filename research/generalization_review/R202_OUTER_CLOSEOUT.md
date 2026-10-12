# R202 image-level outer-assessment closeout

Completed by the user's prepared command on 12 October 2026; verified afterward
from saved files without further inference. **R202 ConvNeXt-V2 Base EMA epoch20
achieved 93.413174% accuracy / 0.901996 macro-F1.** This is a separately reported
single-fold internal image-level outer assessment with natural lesion overlap.
It is not the S83 ensemble's result, strict lesion-independent performance, a
first globally untouched external test, or completed ten-fold cross-validation.

## Frozen method and cohort

- Original R202 accuracy-selected `best.pt`, EMA epoch20. Checkpoint SHA-256:
  `09a7c76b0d6583d1398bf3b31f452a0a51f4cffc6531baa77b9ec227cc204f11`.
- FP32, TF32 disabled, identity only, RGB square224 **bicubic** antialias/ImageNet
  normalization, batch8. No fusion, TTA, calibration or probability correction.
- V2 fold00: 8,111 fitting images, 902 inner-validation images, **1,002 outer
  assessment images**. Zero original image overlap with fitting or selection.
- **389 assessment images share 375 training lesions**. All assessment originals
  are included at their natural class proportions; no favorable subset was chosen.
- This was the first recorded outer inference for this R202 fold/checkpoint.
  Other project studies already used many of these originals. No claim of a
  globally unused dataset or independent external validation is made.
- One complete inference pass saved probabilities before label-based scoring;
  no model/selector/rule change followed. Runtime: **24.33 seconds**, no training.

## Actual scores

| Metric | Result |
|---|---:|
| Accuracy | **93.413174%** |
| Macro precision | 0.897476 |
| Macro recall / balanced accuracy | 0.907846 |
| Macro-F1 | **0.901996** |
| Weighted-F1 | 0.933433 |
| Unweighted negative log likelihood | 0.383568 |
| Correct / incorrect | 936 / 66 |
| Melanoma recall | 77.4775% (86/111) |
| akiec recall | 87.5% (28/32) |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| akiec |0.823529|0.875000|0.848485|32|
| bcc |0.888889|0.923077|0.905660|52|
| bkl |0.906542|0.881818|0.894009|110|
| df |0.923077|1.000000|0.960000|12|
| mel |0.851485|0.774775|0.811321|111|
| nv |0.960236|0.971684|0.965926|671|
| vasc |0.928571|0.928571|0.928571|14|

The confusion matrix is dominated by **21 melanoma images predicted as nevus**,
then 10 nevus images predicted as melanoma. Small df/vasc support limits precision
of those estimates. Raw and row-normalized CSV/PNG/PDF matrices are saved.

## Validation transfer and exposure diagnostics

Original R202 EMA20 inner validation: 93.237251% accuracy / 0.875277 macro-F1.
Outer minus inner validation: **+0.175923 percentage points / +0.026719 macro-F1**.
The estimate transferred reasonably under this image-level protocol. Different
cohort membership means this is descriptive, not a paired treatment-effect test.

| Outer subgroup | Images | Accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|---:|
| Training lesion represented |389|92.030848%|0.915562|82.8571%|
| Training lesion absent |613|94.290375%|0.877339|68.2927%|

Nevus prevalence differs: 200/389 versus 471/613. The higher unseen-group accuracy
does not establish stronger minority-class performance or a causal overlap effect.
The headline result remains the **complete 1,002-image** score.

## S83 stays separate

Frozen S83 six-model + CBAM + Grad-CAM remains **94.011976% / 0.895201 exploratory
validation**, and **87.558217% / 0.807031 on its repeated lesion-separated test
audit**. R202 uses a different model and cohort/protocol; its 93.41% cannot be
assigned to S83, and the difference cannot be attributed solely to overlap.
S83 was trained on 701 of the R202 outer originals and selected using 151 more;
using its weights on this cohort would not be a held-out S83 test.

Permitted paper wording: “ConvNeXt-V2 Base achieved 93.41% accuracy and 0.9020
macro-F1 on a single image-level outer fold, with zero training-image overlap
and naturally occurring training-to-assessment lesion overlap.” Add the
post-development/internal scope, actual split sizes, and separate S83 results.

## Artifacts and verification

[Results package](../../results/image_level_replication/v2/r202_outer_assessment_v1/):
`assessment_metrics.json`, `predictions.csv`, `probabilities_unscored.csv`,
`per_class_metrics.csv`, `lesion_exposure_subgroups.json`, `figures/`, original
freeze/start/completion receipts and added verification/inventory.

Independent CPU review matched all probabilities, class/global scores and
confusion counts, source/checkpoint hashes and registry record. Original four
PNG/PDF figure pairs are preserved. A separate report class-score figure places
its legend outside the bars; its PDF and the normalized matrix were rendered
and visually reviewed. No assessment was rerun, no parameters were updated and
the rejected validation correction remained rejected.

The completed-run guard preserves this result. No automatic further training,
ensemble changes or assessment passes are scheduled. S83 remains the retained
product unless the user makes a separately documented methodology decision.
