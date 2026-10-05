# Saved scores and graphs: quick index

All paths below are relative to the project root.

## Latest provisional exploratory candidate

S53 equal five-model fusion adds S03 EfficientNet-B0 to the four members below, with weight 0.20 each. It achieved **93.6128% accuracy / 0.886857 macro-F1** (1,407 correct / 96 incorrect). The net gain is only two images; it is not established as a material improvement. This cached-probability result has not yet been freshly reproduced with all-five FP32 inference.

Scores, predictions, probabilities, class metrics and PNG/PDF figures: `results/short_screening/s53_equal_five_b0_addition/`. Interpretation: `research/short_screening/S53_EQUAL_FIVE_CLOSEOUT.md`.

## GPU-reproduced four-model reference

**93.4797% accuracy / 0.883680 macro-F1**, 1,405 correct / 98 incorrect on 1,503 exploratory validation images. Equal fusion of ConvNeXt-Tiny, ConvNeXt-Small, DenseNet201 and EfficientNetV2-S using their saved macro-F1-selected checkpoints.

- Original CPU fusion: `results/short_screening/s46_all_f1_checkpoint_fusion/`
- Fresh FP32 identity GPU reproduction (same predicted classes): `results/short_screening/s50_s51_current_ensemble_tta/identity_fp32/`
- Checkpoint paths and hashes: S46 `candidate_manifest.json`; S50/S51 `verification.json` in its parent folder.

Within each result folder:

| File/folder | Contents |
|---|---|
| `validation_metrics.json` | Accuracy, macro precision/recall/F1 and class-wise results |
| `validation_predictions.csv` | Image identities, true/predicted classes and scores |
| `validation_probabilities.csv` | Saved class probabilities |
| `figures/per_class_metrics.csv` | Class-wise precision, recall, F1 and support |
| `figures/confusion_matrix.csv` | Raw confusion matrix |
| `figures/confusion_matrix_normalized.csv` | Normalized confusion matrix |
| `figures/*.png` and `figures/*.pdf` | Confusion matrices, class scores and class support figures |

## Comparisons and rejected checks

- `results/master_experiment_registry.csv`: experiment index and scores.
- `results/short_screening/s46_all_f1_checkpoint_fusion/comparison_figures/`: comparison CSV and PNG/PDF model-comparison plot.
- `results/short_screening/s50_s51_current_ensemble_tta/`: identity control, rejected four-view TTA, verification and summary.
- `results/short_screening/s52_disagreement_resnet_rescue/`: rejected disagreement-only ResNet rule, routing CSV, scores and full figures.
- `results/short_screening/s54_s55_panderm_fusion/`: rejected equal-six PanDerm addition and equal-five B0 replacement; full scores, source manifests, class comparisons and PNG/PDF figures for both.
- `results/short_screening/convnextv2_transfer_v1/s56_frozen_feature_screen/`: failed V2 frozen-feature gate; its 20-epoch S57 run was not launched.
- `results/short_screening/supervised22k_transfer_v1/s58_frozen_feature_screen/`: passed supervised22k frozen-feature gate, scores/predictions and figures.
- `results/short_screening/supervised22k_transfer_v1/s59_convnext_tiny_supervised22k_exploratory_seed42/`: authorized capped fine-tuning run; artifact policy and commands in `research/short_screening/S58_S59_SUPERVISED22K_PLAN.md`.
- S59 is now closed out: best accuracy 90.6853% at epoch20; best macro-F1 0.832974 at epoch18. `research/short_screening/S59_S61_CLOSEOUT.md` records verification and the decision not to extend training.
- `results/short_screening/s60_s61_s59_fusion/`: two rejected S59 addition/replacement ensembles, both 93.0805%, with full metrics, prediction files and figures. S53 remains the provisional highest exploratory accuracy.
- `results/short_screening/s62_s53_error_audit/`: descriptive audit of all 96 S53 errors, six image review sheets, preprocessing comparison, confidence/quality/class CSVs and PNG/PDF figures. Findings and the next bounded calibration proposal are in `research/short_screening/S62_ERROR_AUDIT.md`; no model score changed during this audit.
- `research/short_screening/`: experiment scripts, plans and closeout Markdown files explaining decisions.

## Training and historical evidence

- `results/structured_experiments/`: individual structured model runs, training logs/history, validation metrics/predictions and figures.
- `checkpoints/structured/`: preserved model checkpoints, including best raw accuracy, best macro-F1 and latest checkpoints where produced.
- Training curves belong to training runs. CPU fusion/inference-only experiments have result/comparison figures, not new epoch curves.

## Separate strict and original test results

- `research/FINAL_LOCKED_TEST_RESULTS.md`: original held-out test report.
- `results/final_locked_test/v1/ensemble/test_metrics.json`: original frozen ensemble test metrics, **86.7598% accuracy / 0.794473 macro-F1**.

The 93.48% reference and provisional 93.61% candidate are exploratory validation performance from subsequent development; neither is a new locked-test accuracy. No test evaluation is repeated by these studies.
