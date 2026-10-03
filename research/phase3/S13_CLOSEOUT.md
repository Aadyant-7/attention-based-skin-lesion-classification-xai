# S13 closeout: FP32 control and fixed four-flip TTA

Completed3 October2026. Closeout uses saved outputs only; no GPU inference/training or test images accessed.

## Decision

**Reject this fixed four-flip TTA. Retain S12 as the strongest ensemble methodology.** FP32 identity evaluation gives92.4817% vs saved AMP S12 92.4152%, one extra correct image; this is a numerical control difference, not a substantial method improvement. Future inference comparisons may use this recorded FP32 identity baseline for numerical safety/consistency. No extra flips/weights/combinations were searched.

## Same exploratory validation results

| Method | Accuracy | Macro precision | Macro recall | Macro-F1 | Correct/1503 | Model passes |
|---|---:|---:|---:|---:|---:|---:|
| S06 ConvNeXt | 91.7498% | 0.880435 | 0.853326 | 0.862831 | 1379 | 1 |
| S09 previous triple | 91.7498% | 0.888868 | 0.855425 | 0.869094 | 1379 | 3 |
| S12 B0/ConvNeXt/V2-S | 92.4152% | 0.902493 | 0.858557 | 0.877023 | 1389 | 3 |
| S12 FP32 identity control | 92.4817% | 0.903260 | 0.858699 | 0.877454 | 1390 | 3 |
| S13 four-view TTA | 91.5502% | 0.895956 | 0.837920 | 0.861618 | 1376 | 12 |

FP32 identity versus S12:1 fixed/0 broken, +.06653 percentage points, +.000431 F1. TTA versus S12:10 fixed/23 broken, net13 fewer correct, -.86494 points/-.015405 F1. TTA versus matched FP32 identity:10 fixed/24 broken, net14 fewer correct, -.93147 points/-.015836 F1. Thus FP32 contributes a tiny positive numerical difference; TTA itself worsens accuracy and macro-F1. Runtime167.26 seconds (2.79min), including all12 inference passes/figures; control runtime not separately timed.

## Class-wise control versus TTA

| Class | Support | FP32 P | FP32 R | FP32 F1 | TTA P | TTA R | TTA F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec | 49 | 0.8750 | 0.7143 | 0.7865 | 0.8684 | 0.6735 | 0.7586 |
| bcc | 77 | 0.8391 | 0.9481 | 0.8902 | 0.8295 | 0.9481 | 0.8848 |
| bkl | 165 | 0.9110 | 0.8061 | 0.8553 | 0.8993 | 0.8121 | 0.8535 |
| df | 17 | 1.0000 | 0.8235 | 0.9032 | 1.0000 | 0.7647 | 0.8667 |
| mel | 167 | 0.8408 | 0.7904 | 0.8148 | 0.8267 | 0.7425 | 0.7823 |
| nv | 1007 | 0.9479 | 0.9762 | 0.9618 | 0.9386 | 0.9722 | 0.9551 |
| vasc | 21 | 0.9091 | 0.9524 | 0.9302 | 0.9091 | 0.9524 | 0.9302 |

**akiec:** recall71.43%→67.35%, F1.7865→.7586, two fewer correct. **Melanoma:** recall79.04%→74.25%, F1.8148→.7823, eight fewer correct. FP32 alone leaves akiec and melanoma recall unchanged; its one nv correction reduces a melanoma false positive, slightly increasing melanoma precision/F1. TTA slightly raises bkl recall but lowers its F1; df/nv also regress, vasc unchanged. No justification to retain the12-pass method.

## Confusion matrices

Rows true, columns predicted; order akiec,bcc,bkl,df,mel,nv,vasc.

FP32 identity control:
```
  35    9    4    0    0    1    0
   0   73    0    0    1    3    0
   5    1  133    0    9   17    0
   0    2    1   14    0    0    0
   0    0    3    0  132   32    0
   0    2    5    0   15  983    2
   0    0    0    0    0    1   20
```

Four-view TTA:
```
  33    9    3    0    0    4    0
   0   73    0    0    1    3    0
   5    1  134    0    8   17    0
   0    2    1   13    0    1    0
   0    0    4    0  124   38    1
   0    3    7    0   17  979    1
   0    0    0    0    0    1   20
```

## Verification / paths

- Run: `results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42/`.
-12 per-model/view prediction tables checked for1503 unique manifest identities, correct labels, finite normalized probabilities and consistent argmax.
-Both aggregate probabilities recomputed exactly from fixed parent/view means; weighted probability loss, precision/recall/F1, confusion matrices and supports independently recomputed.
-Metrics/predictions: `validation_metrics.json`, `validation_predictions.csv` (TTA); `validation_metrics_identity_fp32.json`, `validation_predictions_identity_fp32.csv` (control).
-Figures: `figures/` (TTA), `figures_identity_fp32/` (control), each four PNG300dpi/vectorPDF pairs and matrix/classCSV files verified. No training curves/new checkpoints because inference only; retain parent histories/checkpoints.
-Comparison: `results/model_comparison/structured/s13_tta/`, including five-method comparison PNG/PDF/table, class table, aligned predictions and closeout audit.
-Registry: `results/master_experiment_registry.csv`, S13 TTA row plus separately identified FP32-control row; control config/record saved in the run. All unrelated/historical rows unchanged.
-Source signature, original config, split hash and all S03/S06/S10 best checkpoint hashes verified. Raw/test images not opened during closeout.
-Same image-level70/15/15 development split;563 shared train/val lesions disclosed. No independent-lesion or locked-test claim. Fresh strict finalist training later.

## Next

[S14 fixed224/320 ConvNeXt resolution inference](S14_PROPOSAL.md), one frozen candidate only. Prepared CPU checks; GPU approval pending. No additional backbone/CBAM/TTA sweep authorized.

Current planning estimates: Phase3 progress94%; total project76%. Strict confirmation, locked-test evaluation, final XAI/paper remain.
