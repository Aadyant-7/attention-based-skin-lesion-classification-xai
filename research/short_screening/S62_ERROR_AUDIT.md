# S62: S53 validation error audit

Descriptive audit only. The reference remains **93.6128% accuracy / 0.886857 macro-F1**, 1,407 correct / 96 incorrect on the existing 1,503-image exploratory validation cohort. No new model predictions, training, fusion alternatives or locked-test evaluation were performed.

## Where the errors are

| Class | Train images | Validation errors / support | Errors with some member correct | Errors all members miss |
|---|---:|---:|---:|---:|
| akiec | 229 | 14 / 49 | 9 | 5 |
| bcc | 359 | 2 / 77 | 1 | 1 |
| bkl | 769 | 24 / 165 | 19 | 5 |
| df | 81 | 4 / 17 | 3 | 1 |
| mel | 778 | 30 / 167 | 20 | 10 |
| nv | 4,693 | 21 / 1,007 | 16 | 5 |
| vasc | 100 | 1 / 21 | 1 | 0 |

Melanoma, BKL and akiec contribute **68/96 errors**. The largest confusion is melanoma -> nevus (27 images), followed by nevus -> melanoma (14) and BKL -> nevus (13). The imbalance is substantial, but class count alone does not explain success: BCC and vascular-lesion recall are already high despite smaller training support.

**69 errors contain at least one correct member prediction; 27 are missed by all five.** Of the 96 errors, 16 have unanimous wrong votes. This separates current fusion disagreement from shared representation errors. It does not provide an achievable oracle routing rule: choosing the correct member from labels would not be deployable.

Among all validation images, five-member unanimous decisions have 98.61% accuracy (16 errors / 1,153). Decisions with a maximum agreement of three votes have 67.24% accuracy (38 / 116 wrong). Confidence ranks difficult cases reasonably well (error-ranking AUC using negative confidence: 0.8976), but abstaining on difficult cases would change coverage and is not an accuracy improvement on the full cohort.

## Confidence and fusion

These scores describe the exact checkpoints currently in S53, not each model's historical accuracy winner.

| Member | Accuracy | Mean confidence | Mean confidence when wrong | Ten-bin ECE |
|---|---:|---:|---:|---:|
| Tiny, macro-F1 winner | 91.48% | 96.76% | 83.67% | 0.0530 |
| Small, macro-F1 winner | 92.22% | 97.23% | 84.80% | 0.0528 |
| DenseNet201, macro-F1 winner | 89.55% | 94.42% | 80.19% | 0.0549 |
| EfficientNetV2-S, macro-F1 winner | 89.49% | 95.13% | 80.68% | 0.0618 |
| B0, accuracy winner | 86.36% | 91.61% | 73.49% | 0.0535 |

The ensemble is better calibrated overall (ten-bin ECE 0.02595); mean confidence is 66.58% on errors and 92.69% on correct decisions. Only ten errors have ensemble confidence >=0.9; 23 have confidence >=0.8.

The per-model confidence gaps support investigating calibration before adding more weak models. They do not prove calibration is the sole cause of the plateau. Some useful correct member probabilities can lose to confident incorrect probabilities, but the previous fixed fusion/stacking failures show that simply exploiting disagreement is not straightforward.

## Images, labels and preprocessing

- All **1,503 validation images decoded successfully**, all 600x450. Image IDs, class order, source hashes and exact reconstruction of the five-way average were verified. No missing/duplicate validation IDs or changed score ordering were found.
- All five run configs use 224x224 bilinear square resizing and the same ImageNet normalization. There is no observed cross-member preprocessing mismatch.
- Square resizing changes the original 4:3 geometry. This is a consistent design choice, not a discovered implementation bug. An aspect-preserving pipeline is a future hypothesis requiring matched training/inference, not an established fix. Prior resolution/TTA studies do not justify changing inference blindly.
- Six contact sheets covering **all 96 errors** and a side-by-side preprocessing sheet for the top eight confident errors were visually inspected. Hair, dark dermoscope borders, illumination differences, small lesions and occasional edge truncation occur, but many errors have clearly visible images without such artifacts. Visual inspection did not establish one universal quality cause or verify clinical diagnoses.
- Brightness medians are similar for correct and wrong cases (155.19 vs 154.47 on a 0–255 scale). The error group's median Laplacian variance is higher (61.06 vs 41.32), not uniformly lower; texture/hair also affect this proxy. Within-class quality tables show mixed patterns, so calling blur the cause would be unsupported.
- **89/96 errors have histology-derived labels.** That argues against assuming weak label provenance explains most errors. It does not certify every label. Diagnostic source is strongly confounded with class (all melanoma/akiec/BCC validation labels are histology-derived); do not infer that histology itself causes errors or relabel cases from these thumbnails.
- Images sharing a training lesion ID: 92.79% accuracy (43 / 596 wrong); remaining exploratory validation images: 94.16% (53 / 907 wrong). These descriptive groups have different case composition; neither is an independent strict confirmation result.

## Decision and next bounded check

**Do not launch another backbone or extend S59 based on this audit.** Retain S53 as the provisional reference, and keep all original strict/test results intact.

The next justified cheap experiment is **one cross-fitted, per-member scalar temperature-calibration study**:

1. Reuse only these exact five checkpoints' saved probability vectors.
2. Use five fixed stratified lesion-group meta folds; fit each member's single temperature on the other folds by negative log-likelihood, not accuracy or ensemble weights.
3. Apply it only to each held-out meta fold and average the calibrated five probabilities equally. No class-specific thresholds, learned voting weights, alternative checkpoints or temperature grid search.
4. Verify each member's argmax remains unchanged; calibration adjusts confidence, not standalone class predictions. Inspect accuracy, macro-F1, melanoma/akiec recall, NLL and reliability.
5. Require a predeclared material gate of at least +0.005 absolute accuracy (at least eight net images here) with no macro-F1 or melanoma-recall decline. Preserve a failed result and stop; do not tune until the gate passes.

Temperature scaling is supported as a calibration method by [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html); this paper does not establish an accuracy gain for our ensemble. Cross-fitting protects the calibration fit from its own holdout labels; it does **not** erase prior checkpoint/method selection on this full validation cohort. Report any outcome as post-test exploratory development, not a new independent test score. No calibration experiment was run in this audit.

If this bounded check fails, the audit does not yet justify further GPU expenditure. A larger change would need a separately substantiated training/preprocessing hypothesis; do not turn these quality correlations into an automatic augmentation package.

## Artifacts

`results/short_screening/s62_s53_error_audit/` contains:

- `errors_96.csv`, `all_validation_audit.csv`: case-level errors, probabilities/confidence, votes, component correctness, metadata and quality proxies.
- `class_error_summary.csv`, `error_confusions.csv`, `pairwise_error_overlap.csv`: class and error-complementarity summaries.
- `model_confidence.csv`, `ensemble_reliability.csv`: calibration descriptions.
- `within_class_quality.csv`, `descriptive_strata.csv`: quality/source/overlap comparisons, not causal tests.
- `preprocessing_recipes.json`, `source_verification.json`: exact member configs, source/image hashes and split checks.
- `figures/`: PNG/PDF confusion-pattern, confidence/reliability and error-decomposition graphs.
- `contact_sheets/errors_01.png` through `errors_06.png`, plus `preprocessing_top8.png`: visual review sheets.

All image access was restricted to validation IDs after excluding original locked-test identities. No original result, prediction, label, checkpoint or image was modified. Original held-out test accuracy remains **86.7598%**.
