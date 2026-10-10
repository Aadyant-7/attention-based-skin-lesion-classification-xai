# Final exploratory ensemble: S83 frozen

10 October2026. The final bounded S97 warm-start is closed. No more training or validation searches are planned. Its final test audit is now complete:87.558217% /0.807031200 macro-F1. See `research/FINAL_S83_TEST_AUDIT.md`; method unchanged afterward.

## Last experiment and decision

S97 stopped at cumulative epoch51:33 inherited +18 newly trained. Both best accuracy/F1 remained inherited epoch33:93.013972% /0.877434398. New-only best accuracy was92.681304% at41; new-only best F1 was0.875961656 at49. Final51 was92.215569% /0.874496102. Patience exhausted after18 epochs without meaningful improvement, following four child LR reductions. One AMP gradient overflow at43 was skipped/backed off and documented; all committed states remained finite.

CPU audit verified selected/latest checkpoints, full recovery state, source hashes, predictions/probabilities, metrics, class scores, raw/normalized confusion matrices and annotated PNG/PDF training curves. Parent evidence is preserved.

The one predeclared S98 equal-six replacement reached93.945442% /0.893189677, gained2 and lost3 versus S83. Because S97's winner remained inherited33, this simply reproduces the already-known S82 method; it is not a new training improvement. Retain original S83 and its original CBAM F1-selected35 checkpoint.

## Exact final six-model method

| Slot | Model | Run | Selected epoch | Selection | Weight |
|---|---|---|---:|---|---|
| 1 | convnext tiny | S06 | 18 | macro_f1 | 1/6 |
| 2 | convnext small | S18 | 17 | macro_f1 | 1/6 |
| 3 | densenet201 | S15 | 14 | macro_f1 | 1/6 |
| 4 | efficientnet v2 s | S10 | 15 | macro_f1 | 1/6 |
| 5 | efficientnet b0 | S03 | 20 | accuracy | 1/6 |
| 6 | convnext tiny + CBAM | S79 | 35 | macro_f1 | 1/6 |

All components originated from the listed ImageNet1k torchvision weights and were fine-tuned on the exploratory training partition. Ensemble inference:224x224 square bilinear RGB/ImageNet normalization; FP32; identity only; no TF32, TTA, multi-resolution, inference augmentation, calibration or metadata. Average the six softmax probability vectors equally, then argmax in class order akiec,bcc,bkl,df,mel,nv,vasc. This is inference fusion of separately trained models, not joint ensemble training.

Observed exploratory validation result: **94.011976% accuracy /0.895200837 macro-F1**, macro precision0.922321360, macro recall0.874217187. Correct1413/1503; incorrect90. Melanoma recall82.035928%; akiec recall71.428571%. These scores belong to the unchanged S83 checkpoints, not S97. The original predeclared material-gain threshold versus S53 was not met (+6 net correct versus required8); S83 is selected here as the numerical best, not evidence of a large or independently confirmed gain.

## Saved sources and artifacts

- `results/final_exploratory_freeze/v1/frozen_method.json`: exact six checkpoint paths/hashes, selectors, epochs, pretraining, preprocessing, weights, class order and validation provenance.
- `results/short_screening/s83_cbam_f1_addition_cpu/`: final validation scores, prediction probabilities, class scores, confusion/comparison PNG/PDF figures and original source plan.
- `results/short_screening/final_convnext_warmstart_v1/s97_convnext_tiny_cbam_best_warmstart_exploratory_seed42/`: completed S97 summary, verification, ancestry/child histories, LR history, raw accuracy/F1/final predictions/metrics and figures.
- `checkpoints/short_screening/final_convnext_warmstart_v1/s97_convnext_tiny_cbam_best_warmstart_exploratory_seed42/`: S97 best/F1/latest and recovery snapshots, preserved locally.
- `results/short_screening/s98_final_warmstart_replacement_cpu/`: the single fixed comparison, source hashes, predictions, class/confusion/comparison figures and verification.
- `results/master_experiment_registry.csv`: completed experiment index; historical rows/checkpoints retained.

## Original pre-audit plan (audit step now completed)

Freeze these exact six checkpoints and rules → after explicit user approval, one final recorded audit on the previously evaluated1503-image original test cohort → save final test metrics/predictions/class scores/confusion figures without any subsequent method tuning → Grad-CAM/XAI for the exact frozen components → final tables/report/paper.

The94.01% is exploratory **validation**, not test accuracy. Image-level development can share related lesions; validation was reused extensively. The original test cohort was already evaluated in earlier stages, so the next result is a **repeated post-development audit**, not a first untouched independent test. Do not assume validation94.01% implies similar test accuracy. XAI explains decisions and does not improve predicted labels or scores. Existing XAI from earlier five-model methods cannot be relabelled as explanations of this six-model method without generating matching explanations.
