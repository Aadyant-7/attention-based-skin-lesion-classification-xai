# Controlled ensemble evidence

This is a navigation folder. Fusion packages stay at their recorded paths; do not duplicate or regenerate metrics during cleanup.

| Ensemble evidence | Actual location |
|---|---|
| Early matched S04 and S07–S13 fusions/TTA comparisons | [../model_comparison/structured/](../model_comparison/structured/) and matching run packages. |
| Retained exploratory S83 equal-six | [../short_screening/s83_cbam_f1_addition_cpu/](../short_screening/s83_cbam_f1_addition_cpu/) — **94.011976% / 0.895200837 macro-F1**. |
| Exact S83 checkpoint/weight/preprocessing freeze | [../final_exploratory_freeze/v1/frozen_method.json](../final_exploratory_freeze/v1/frozen_method.json). |
| Latest R204/R205 Base addition/replacement comparisons | [../structured_experiments/convnext_base_original_v1/ensemble_comparison/](../structured_experiments/convnext_base_original_v1/ensemble_comparison/) — neither improves S83 accuracy. |
| Separate strict S31 equal-three | [../final_strict/v1/ensemble/](../final_strict/v1/ensemble/) — **90.1530% strict validation / 0.843486 macro-F1**. |

S83 averages probabilities equally from ConvNeXt-Tiny, ConvNeXt-Small, DenseNet201, EfficientNetV2-S, EfficientNet-B0 and ConvNeXt-Tiny + CBAM. Inference is FP32, identity view, 224 pixels; weights are **1/6 each**. Member checkpoint selectors are explicitly frozen. This is inference fusion of separately trained networks, not joint ensemble training.

These scores are validation observations. The [S83 test audit](../../research/FINAL_S83_TEST_AUDIT.md) and original strict held-out evaluation remain separate. See the [reporting evidence index](../../reporting/evidence/INDEX.md) for comparisons and protocol labels.
