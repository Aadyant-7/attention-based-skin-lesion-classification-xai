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
- `research/short_screening/`: experiment scripts, plans and closeout Markdown files explaining decisions.

## Training and historical evidence

- `results/structured_experiments/`: individual structured model runs, training logs/history, validation metrics/predictions and figures.
- `checkpoints/structured/`: preserved model checkpoints, including best raw accuracy, best macro-F1 and latest checkpoints where produced.
- Training curves belong to training runs. CPU fusion/inference-only experiments have result/comparison figures, not new epoch curves.

## Separate strict and original test results

- `research/FINAL_LOCKED_TEST_RESULTS.md`: original held-out test report.
- `results/final_locked_test/v1/ensemble/test_metrics.json`: original frozen ensemble test metrics, **86.7598% accuracy / 0.794473 macro-F1**.

The 93.48% reference and provisional 93.61% candidate are exploratory validation performance from subsequent development; neither is a new locked-test accuracy. No test evaluation is repeated by these studies.
