# Research reset — 5 October 2026

## What has already been achieved

All rows below are the same exploratory validation protocol. All used ImageNet transfer learning; transfer learning is already implemented. CBAM is an ablation, not a universal improvement.

| Model | Validation accuracy | Macro-F1 |
|---|---:|---:|
| mobilenet_v3_large | 86.03% | 0.7861 |
| efficientnet_b0 | 86.36% | 0.7707 |
| efficientnet_b0 + CBAM | 86.36% | 0.7693 |
| efficientnet_v2_s | 89.49% | 0.8232 |
| densenet201 | 89.62% | 0.8075 |
| convnext_tiny | 91.75% | 0.8628 |
| convnext_small | 92.22% | 0.8549 |
| resnet101 | 88.09% | 0.7920 |

Frozen exploratory equal B0/ConvNeXt-Tiny/V2-S ensemble: 92.4817% / .877454 macro-F1. Correction after reviewing the complete final-selection table: highest prior numerical exploratory alternative was S27 at 92.7478% / .866355 macro-F1, not the initially quoted92.6148%.

New CPU-only S42 equal ConvNeXt-Tiny/ConvNeXt-Small/DenseNet201/EfficientNetV2-S fusion reached93.1470% / .879744 on the SAME exploratory validation cohort. This is now the highest observed exploratory score, a modest improvement, not a new held-out test result. See `research/short_screening/S42_EXPLORATORY_RESULTS.md` for verification and limitations. Original strict/test results remain unchanged.

Fresh strict ensemble validation: 90.1530% / .843486. Original one-time locked test: 86.7598% / .794473. These are different cohorts/protocols, not interchangeable estimates.

## Verified today

CPU-only audit: all 8,512 development images pass integrity checks, all RGB 600x450, no byte-identical train/validation duplicates. Manifest labels match raw development metadata; zero lesion overlap is verified from partition identities. Three strict constituent prediction files align exactly with validation IDs/labels; class probability order, argmax labels and recomputed accuracy/F1 match saved metrics. No inference or test images/labels were accessed. This does not exclude perceptual duplicates, clinically uncertain labels or preprocessing limitations.

## Evidence and limits

- Latest paired fine-tuning reached about 99.8% training accuracy but 88–89% validation accuracy. This is evidence of a generalization gap, not insufficient ability to fit training data. Training accuracy is from augmented/repeated sampling and is not a matched inference evaluation.
- Both accuracy winners retained the S29 parent. Targeted sampling did not outperform its control; do not repeat or extend it.
- Aggressive recipe changed weighting, head, normalization, augmentation, optimization and loss together. It underperformed; its exact causal contribution cannot be separated retrospectively.
- MEL/BKL account for 97/148 strict-validation errors. The saved development review contrasts high-confidence errors with correct controls; selection is deliberately diagnostic and not representative prevalence.
- 224-square resizing changes the 600x450 aspect ratio and reduces detail. This is a candidate input limitation, not a demonstrated explanation; it affects correct examples too.
- A repeatedly used validation cohort can become optimistic. Original test is now exposed; further work is post-test development and cannot yield another pristine held-out claim from the same test.

## Next decision, before more compute

Visual review completed for the 24 selected examples: several confident MEL/BKL errors occupy a relatively small part of the frame, while several correctly classified MEL examples fill more of it. Hair, borders and acquisition variation occur in the examples. This is a selected montage, not a measured population association or a clinical diagnosis. It supplies a lesion-scale/input-resolution hypothesis to measure on development data; it does not justify automatic cropping, label edits, or a promised accuracy gain.

1. Review the saved development contact sheet and training examples for lesion visibility, hair/borders, crop/scale issues and annotation ambiguity. Do not relabel clinical classes based on our guesses.
2. If a concrete input issue is found, define ONE controlled training-from-ImageNet comparison with the proven ConvNeXt recipe; test the identified representation/generalization change from early training, rather than late fine-tuning a nearly memorized model. Fix the budget and preserve the control. No run prepared or authorized by this document.
3. If the review supplies no defensible intervention, stop the performance loop and present the completed backbone/ensemble research and its actual limitations. CPU audits cannot establish future training gains.
4. Preserve original test results and all checkpoints. No automatic queue, additional backbone, fusion search or test rerun.

## Files

- Existing model table: `results/model_comparison/final_architecture_selection/model_vs_accuracy.csv`
- New input audit: `results/short_screening/reset_evidence_audit/input_and_prediction_audit.json`
- Development visual review: `results/short_screening/reset_evidence_audit/development_error_review.png`
- Review identities: `results/short_screening/reset_evidence_audit/development_visual_review_index.csv`
- Detailed prior diagnosis: `research/aggressive/ROOT_CAUSE_AND_RECOVERY_PLAN.md`
- Original held-out report: `research/FINAL_LOCKED_TEST_RESULTS.md`
