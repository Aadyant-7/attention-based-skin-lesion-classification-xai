# S02 closeout — MobileNetV3-Large exploratory screening

Completed 3 October 2026;20 epochs, **epoch16 selected by both accuracy and macro-F1**. ID `s02_mobilenet_v3_large_none_exploratory_seed42`.

| Validation metric | Value |
|---|---:|
| Accuracy | **86.02794411177644%** (1,293/1,503) |
| Macro precision |0.7949701703544128 |
| Macro recall / balanced accuracy |0.780099667659236 |
| Macro-F1 |**0.7861111676525011** |
| Weighted loss |0.59081971124727 |
| Melanoma recall |69.46107784431138% (116/167) |
| Runtime |896.62 seconds / **14.94 minutes** |
| Parameters / peak allocated VRAM |2,978,679 /475.71MiB (not total reservation) |

## Protocol and comparison

Existing `data/splits/exploratory/image_level_dev_v1.csv`,SHA `75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb`:7,009 train/1,503 validation/1,503 locked test (~70/15/15).563 shared train/val lesions: **exploratory, not lesion-independent performance**. Test identities unchanged; no test loader/evaluation.

ImageNet V2 MobileNetV3-Large, native SE, no added CBAM; common GAP/dropout/linear head,224 input,flip,training-only sqrt weights,AdamW,batch16/accumulation2,AMP,seed42. Accuracy scheduler/early stopping plus independent macro-F1 winner. Stop reason:20-epoch cap. Initial download blocked before training; official hash-verified weights downloaded and same config retried. Failed launch evidence retained under `results/audit/s02_pretraining_download_failed_20261003/`.

| Same exploratory validation cohort | Accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|
| **S02 MobileNetV3-Large, single model** |**86.03%** |**0.7861** |**69.46%** |
| Historical weighted B0+CBAM, single model |86.23% |0.7808 |67.66% |
| Historical strong-augmentation B0+CBAM |85.30% |0.7657 |68.26% |
| Historical B0+CBAM, four-stream ensemble |88.16% |0.8280 |68.86% |
| Historical B0+CBAM + PanDerm, eight-stream ensemble |90.75% |0.8573 |70.66% |

Historical registry manifest hashes and class supports match. Versus weighted single B0+CBAM:3 fewer correct predictions/**-0.1996 accuracy percentage points**, **+0.005348 macro-F1**, **+1.7964 melanoma-recall points**. Small differences establish neither superiority nor significance. Heads,attention,batches,selection and ensemble costs differ: descriptive recipe context, not controlled architecture ranking. Historical repeated validation selection can inflate ensemble maxima.

Context table/graph: `results/model_comparison/structured/s02_exploratory_context/`. Controlled structured table `results/model_comparison/structured/exploratory_screening_v1_seed42/comparison.csv` contains only S02. Strict S01 stays separate.

## Interpretation and next step

MobileNet is competitive with the historical single model; it has not exceeded the ensemble maximum. LR reduction before13 helped; minimum val loss at13, accuracy/F1 peak16. Final train accuracy96.25% versus validation85.16%, with rising late val loss: growing generalization gap. Logged optimizer updates4,375/4,380 imply five AMP skips, preserved as measured.

Melanoma errors:35→nv,11→bkl; nv45→mel; bkl22→nv,17→mel. Melanoma F1=.6667, bkl recall=.7091. df/vasc supports17/21 make estimates unstable. Complementarity requires aligned predictions, not aggregate-score differences.

Recommend [S03 matched exploratory B0 control](S03_PROPOSAL.md): fair common-recipe architecture comparison, a baseline for CBAM and aligned predictions for bounded fusion. No unchanged extension, automatic model queue or next GPU run.

## Saved and verified

- `results/structured_experiments/s02_mobilenet_v3_large_none_exploratory_seed42/`: config/environment/pretraining/record, log/progress,history,summary,primary and macro-F1 metrics/predictions,closeout verification. Both prediction tables contain1,503 IDs/labels/seven probabilities.
- `figures/`: seven PNG/PDF curve/matrix/class/support pairs and three CSVs; `figures_macro_f1/`: secondary matrix/class/support figures/CSVs. Both winners have identical epoch16 model weights.
- `checkpoints/structured/s02_mobilenet_v3_large_none_exploratory_seed42/`: `best.pt`,`best_macro_f1.pt`,`latest.pt` (epoch20/full state). CPU loads verify weights,epochs,metrics,hashes/history/probability-derived matrices. Checkpoints/logs remain local; metrics/predictions/figures committed.
- Master registry completed; all473 historical rows unchanged. Preservation checks pass; accuracy curve and normalized matrix visually checked.

Current Phase Progress: 65%
Total Project Progress: 60%
