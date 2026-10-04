# Budget decision and CPU-only screening

Long queue stopped; ResNet101 cancelled before launch. No test inference, raw-image loading, weight search or new training. DenseNet is incomplete, not a full-run failure.

DenseNet best saved accuracy: 82.3686% at epoch 18; best macro-F1 0.682927 at epoch 11; committed epochs 19.

| Method | Accuracy | Macro-F1 | Reference errors fixed | Correct lost |
|---|---:|---:|---:|---:|
| strict_reference | 90.1530% | 0.843486 | 0 | 0 |
| reference_plus_b3_equal | 89.4212% | 0.822037 | 16 | 27 |
| reference_plus_dense_equal | 89.2881% | 0.810033 | 15 | 28 |
| reference_plus_b3_dense_equal | 89.0220% | 0.817775 | 17 | 34 |

These are post-test development comparisons, not new held-out test results. The reference is itself the equal B0/ConvNeXt-Tiny/EfficientNetV2-S ensemble; added-model comparisons average at the ensemble level.
