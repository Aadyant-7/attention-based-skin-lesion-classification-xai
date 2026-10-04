# Final strict lesion-disjoint validation results

Architecture permanently closed: B0 + ConvNeXt-Tiny + EfficientNetV2-S, equal 1/3 averaging, FP32 identity, 224 pixels. Fresh ImageNet initialization. Locked test NOT evaluated.

| Model | Best epoch | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---|---:|---:|---:|---:|
| efficientnet_b0 | 19 | 85.8283% | 0.741034 | 0.753598 | 0.744897 |
| convnext_tiny | 33 | 89.0220% | 0.815844 | 0.814441 | 0.814678 |
| efficientnet_v2_s | 24 | 85.4291% | 0.785217 | 0.712033 | 0.742685 |
| Frozen equal ensemble | individual accuracy winners | 90.1530% | 0.871298 | 0.822827 | 0.843486 |

## Final ensemble class performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| akiec | 0.804348 | 0.740000 | 0.770833 | 50 |
| bcc | 0.819277 | 0.883117 | 0.850000 | 77 |
| bkl | 0.858065 | 0.806061 | 0.831250 | 165 |
| df | 1.000000 | 0.888889 | 0.941176 | 18 |
| mel | 0.834711 | 0.608434 | 0.703833 | 166 |
| nv | 0.925542 | 0.976143 | 0.950169 | 1006 |
| vasc | 0.857143 | 0.857143 | 0.857143 | 21 |

Correct: 1355; incorrect: 148. Melanoma recall 0.608434; akiec recall 0.740000.

Strongest standalone: convnext_tiny. Ensemble accuracy change +1.1311 percentage points; macro-F1 change +0.028808; predictions gained/lost 48/31.

## Exploratory versus strict

Exploratory reference: 92.4817% / 0.877454. Strict: 90.1530% / 0.843486. Accuracy difference -2.3287 percentage points; macro-F1 difference -0.033968.

Different protocols are not interchangeable: exploratory train/validation share 563 lesions affecting 596 validation images; strict partitions share zero lesions. Validation cohorts and training duration/numerical recipe also differ. This gap cannot be attributed solely to leakage, and lower strict performance is not evidence of a failed experiment. Repeated exploratory selection adds optimism; this remains validation evidence, not test performance.

Class-wise changes: `results/final_strict/v1/exploratory_vs_strict_class_changes.csv`. Raw confusion matrix (rows true, columns predicted; ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']):

```text
[[ 37   6   3   0   1   3   0]
 [  0  68   1   0   2   4   2]
 [  6   1 133   0   5  20   0]
 [  0   0   0  16   0   2   0]
 [  3   1  11   0 101  49   1]
 [  0   6   7   0  11 982   0]
 [  0   1   0   0   1   1  18]]
```

Saved results: `results/final_strict/v1/`; individual runs `results/structured_experiments/s28_*`, `s29_*`, `s30_*`; checkpoints `checkpoints/structured/` under matching IDs. All accuracy/F1/latest states retained. Ensemble checkpoint identities are in `ensemble/frozen_ensemble.json`. Exploratory table remains `research/MODEL_VS_ACCURACY.md`; strict table is separate.

STOP: review strict validation before one authorized locked-test evaluation, then Grad-CAM/XAI. No further architecture, weight, TTA or resolution search.
