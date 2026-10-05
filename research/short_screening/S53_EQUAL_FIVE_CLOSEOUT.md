# S53: fixed equal five-model B0 addition

One CPU-only comparison added the preserved S03 EfficientNet-B0 accuracy-selected checkpoint to the four S46 macro-F1-selected members: ConvNeXt-Tiny, ConvNeXt-Small, DenseNet201 and EfficientNetV2-S. All five probabilities received weight 0.20. No weights, epochs or inference views were searched.

Cheap error screening of unused saved models found that B0 fixed 31 of the reference's 98 errors, versus MobileNet 28, B0+CBAM 29 and ResNet101 28. This label-based candidate selection used the existing development validation set and is not independent evidence.

| Method | Exploratory accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|
| S46 / S50 four-model reference | 93.4797% | 0.883680 | 79.0419% |
| S53 equal five including B0 | 93.6128% | 0.886857 | 82.0359% |

S53 gained 12 correct predictions and lost 10: net +2, or +0.1331 percentage points (1,407 correct / 96 incorrect of 1,503 images). The directional adoption gate passed. However, this does not establish a material or statistically reliable improvement. Retain S53 as a provisional development candidate and preserve the independently GPU-reproduced S50 four-model reference; do not automatically replace the frozen original strict/test methodology.

The script independently recomputed accuracy, macro-F1 and the confusion matrix from its saved predictions. Image IDs, class order, source hashes, probability normalization and absence of locked-test images were checked. S53 uses cached probabilities with their existing inference precision; fresh all-five FP32 inference has not been performed. No GPU work or locked-test evaluation occurred.

Artifacts: `results/short_screening/s53_equal_five_b0_addition/` includes the predeclared plan, source/checkpoint hashes, summary, metrics, predictions, probabilities, class-wise scores, raw/normalized confusion matrices and PNG/PDF figures/comparisons. The master registry records S53. Original held-out test accuracy remains 86.7598%.
