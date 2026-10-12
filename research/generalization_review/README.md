# Pre-test generalization review

12 October 2026. User asked whether the 94.01% S83 validation versus 87.56% audit
gap reflects overfitting, and requested development checks before another test.
This review uses **already-saved validation predictions and histories only**.
No training or assessment inference has run; S83/XAI remains unchanged.

## Findings

| Method and validation subgroup | Images | Accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|---:|
| S83, all original exploratory validation |1503|94.0120%|0.895201|82.04%|
| S83, lesion represented in training |596|92.6174%|0.929541|82.46%|
| S83, lesion absent from training |907|94.9283%|0.845247|81.13%|
| R202 EMA20, all inner validation |902|93.2373%|0.875277|83.00%|
| R202 EMA20, lesion represented in training |353|92.3513%|0.941453|86.49%|
| R202 EMA20, lesion absent from training |549|93.8069%|0.792605|73.08%|

The unseen S83 subgroup is 76.19% nevus, versus 53.02% in the seen group. R202's
corresponding fractions are 78.69% and 48.73%. Higher unseen-group accuracy is
therefore not evidence of better minority-class or new-lesion performance.
Class-standardized S83 accuracy is 94.0227% seen versus 93.4050% unseen. This
adjusts prevalence only; it does not control image difficulty, within-lesion
correlation, or earlier checkpoint/model selection.

The 94→87.56 gap alone cannot establish one cause. The development/test protocols
differ, validation was reused extensively, and late training can overfit.
Existing S31's strictly separated validation/test also had a gap, so overlap
cannot explain every performance difference. No test labels or pixels were read
to calculate these development diagnostics.

R202 training accuracy increased from 99.396% at 20 to 99.963% at 50. EMA validation
loss worsened from 0.42569 to 0.72036, while accuracy declined from 93.2373% to
92.4612%. Its **original accuracy-selected best.pt is already EMA epoch20**,
not the final weights. That earlier selection is preserved. Low training loss
does not justify extending this run.

## One bounded correction checked and rejected

The R202 loss uses training-derived square-root class weights. One predeclared
CPU check removed the idealized weighted-CE class tilt: divide each saved class
probability by its recorded training loss weight and renormalize. No exponent,
temperature, class threshold, checkpoint or weight grid was searched.

| Rule on the same 902 validation images | Accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|
| Original EMA20 |93.2373%|0.875277|83%|
| Fixed inverse-class-weight correction |92.9047%|0.882148|80%|

It gained 1 correct prediction and lost 4. It failed the predeclared minimum
0.5 percentage-point accuracy gain and nondecreasing melanoma-recall requirements.
**Rejected; no trained weights or original predictions were changed.** This is
a negative development diagnostic, not a new test score.

## Recommended lowest-cost next step

Preserve S83 as the final ensemble reference. For the user's separate
overlap-allowed assessment, use the **existing R202 EMA20** unchanged rather than
retraining all six S83 members. CPU preparation has authenticated its checkpoint,
original launch signature, class order and preprocessing and verified state-load
compatibility. It decoded no assessment images and read no assessment labels.

The fixed V2 fold00 has 8111 training, 902 inner-validation and 1002 assessment
images. Assessment shares **zero original image IDs** with fitting/selection;
389 assessment images share 375 lesions with training. This is one internal
post-development image-level assessment, not S83 performance, strict
lesion-independent performance, a fresh external test or completed ten-fold CV.
Earlier project models trained on some of these originals, so S83 cannot join
this evaluation.

Use FP32/no TF32, identity only, 224-square **bicubic** RGB/ImageNet preprocessing
(R202's original recipe), no fusion/TTA/calibration/correction. One complete
probability pass precedes label-based scoring. Report the entire 1002-image
cohort; exposure subgroups are descriptive and cannot replace the headline score.
No subsequent tuning on assessment labels or automatic extra run is scheduled.

Estimated assessment cost: approximately 1–3 minutes on the RTX 4060, batch 8,
within 8GB VRAM. No training is involved. **Prepared, not launched.**

Start from the existing project root after the evaluation decision is approved:

```powershell
Set-Location 'E:\Minor Project\Project Development'
& '.\.venv\Scripts\python.exe' -B -m research.generalization_review.evaluate_r202 --evaluate
```

Manual monitor in another PowerShell window:

```powershell
Get-Content 'E:\Minor Project\Project Development\results\image_level_replication\v2\r202_outer_assessment_v1\assessment.log' -Tail 20 -Wait
```

Prepared freeze and future outputs:
`results/image_level_replication/v2/r202_outer_assessment_v1/`.
Exact existing checkpoint:
`checkpoints/image_level_replication/v2/r202_convnextv2_base_fold00_seed42/best.pt`.
The completed-run guard prevents repeat inference; an interrupted attempt must
be inspected rather than blindly rerun.

## If genuine new-lesion improvement is requested later

Changing learning rates/dropout/loss settings now does not repair saved weights;
it requires a new fit. The defensible next development trial would be **one
fresh ConvNeXt-V2 Base with a lesion-disjoint inner train/validation split from
the 9013-image V2 development pool**, keeping the 1002 outer identities reserved.
Fresh external pretrained initialization is required because R202's old
training would contaminate a newly drawn validation set. Keep a bounded 20–30
epoch budget, modest augmentation/class weighting, EMA and meaningful stopping;
select on grouped validation, macro-F1/class recall and accuracy together.
This is a conditional recommendation, not a prepared/authorized training run,
and has no guaranteed test gain. No six-model retraining is recommended now.

## Saved evidence

- [Diagnostic audit](../../results/diagnostics/generalization_review_2026-10-12/audit.json),
  [subgroup table](../../results/diagnostics/generalization_review_2026-10-12/validation_subgroups.csv),
  [class scores](../../results/diagnostics/generalization_review_2026-10-12/validation_subgroup_classes.csv).
- [Rejected rule/decision](../../results/diagnostics/generalization_review_2026-10-12/r202_weight_bias_check/summary.json),
  [predeclared rule](../../results/diagnostics/generalization_review_2026-10-12/r202_weight_bias_check/plan.json).
- [Prepared assessment freeze](../../results/image_level_replication/v2/r202_outer_assessment_v1/frozen_assessment.json).
- [Original R202 closeout](../image_level_replication/v2/R202_CLOSEOUT.md),
  [retained S83 product](../../reporting/FINAL_PRODUCT_HANDOFF.md).
