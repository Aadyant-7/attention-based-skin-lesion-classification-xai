# S76: learned fusion of the actual trained features

Completed; rejected. Existing exploratory development split: 7,009 training and 1,503 validation images. Original test images and labels were not loaded.

| Method | Accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|
| Fresh FP32 equal-five control | 93.6128% | 0.886857 | 82.0359% |
| Concatenated trained features + fixed RBF SVM | 91.3506% | 0.850417 | 72.4551% |

The new control reproduced every saved S53 predicted label. Fusion gained 10 correct predictions but lost 44 (net -34); it failed the predefined material-improvement gate. Retain S53. This rejects this fixed feature/classifier recipe, not every possible feature-fusion method.

Features came from the five actual skin-fine-tuned S53 checkpoints (6,016 dimensions total), with training-only scaling, block normalization and one fixed class-weighted RBF classifier. No backbone updates or tuning search occurred. This directly addressed a limitation of earlier CPU screens, which used generic ImageNet features rather than these trained representations. Frozen GPU extraction took 308 seconds. Source hashes, cache hashes, reloaded classifier predictions and all 565 prior registry rows were verified unchanged; two new registry rows were added.

Evidence: `results/short_screening/s76_trained_feature_fusion/` contains the plan, metrics, predictions/probabilities, class scores, confusion matrices, comparison PNG/PDF figures, gain/loss analysis and verification. The local fitted classifier is `checkpoints/short_screening/s76_trained_feature_fusion/feature_fusion.joblib`; caches remain under `.cache/s76_trained_feature_fusion/`.

Research context: [feature versus decision fusion](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1478688/full) motivates checking internal representations but does not guarantee an advantage or supply a comparable accuracy target. All scores here are repeatedly used exploratory validation, not independent test estimates.
