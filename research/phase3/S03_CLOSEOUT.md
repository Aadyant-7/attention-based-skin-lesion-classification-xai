# S03 closeout — matched exploratory EfficientNet-B0 control

Completed 3 October 2026. ID `s03_efficientnet_b0_none_exploratory_seed42`. **20 epochs; best accuracy and macro-F1 both at epoch 20. Final and selected metrics are identical.**

| Metric | S03 selected/final |
|---|---:|
| Accuracy | **86.3606121091151%** (1,298/1,503) |
| Macro precision | 0.7534286077687018 |
| Macro recall / balanced accuracy | 0.7916330829131726 |
| Macro-F1 | **0.7706600163648087** |
| Weighted loss | 0.6304686558695167 |
| Melanoma recall | 70.05988023952096% (117/167) |
| Runtime | 938.30 seconds / **15.64 minutes** |
| Parameters | 4,016,515 |
| Peak allocated VRAM | 851.87 MiB; not total GPU reservation |

## Matched S02/S03 comparison

Same exploratory manifest SHA `75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb`, 7,009 train / 1,503 validation / 1,503 locked test (~70/15/15). Train/validation share 563 lesions; **not lesion-independent performance**. Test images were never loaded. Identical common head, input, augmentation, training-only sqrt weights, optimizer, batch/accumulation, selection, scheduler policy, seed and 20-epoch cap. Only backbone/official pretrained specification change, besides run ID/question. B0 V1 versus MobileNet V2 pretraining is part of the compared model package, not an isolated topology effect.

| Accuracy-selected model | Epoch | Accuracy | Macro-F1 | Macro recall | Mel recall | Time |
|---|---:|---:|---:|---:|---:|---:|
| S02 MobileNetV3-Large | 16 | 86.03% | **0.7861** | 0.7801 | 69.46% | 14.94 min |
| S03 EfficientNet-B0 | 20 | **86.36%** | 0.7707 | **0.7916** | **70.06%** | 15.64 min |

B0 has **five more correct images / +0.3327 accuracy percentage points**, **-0.015451 macro-F1**, and +0.5988 melanoma-recall points. One seed and validation-selected checkpoints do not establish significance. B0 is the accuracy leader of the structured single-model screening; MobileNet remains the macro-F1 leader and smaller alternative. B0 used about 35% more parameters and 79% more allocated VRAM, with measured total runtime about 4.65% higher on these runs; runtime is not a hardware-controlled speed benchmark.

| Class (support) | S02 F1 | S03 F1 | S03 recall |
|---|---:|---:|---:|
| akiec (49) | 0.7292 | 0.6458 | 0.6327 |
| bcc (77) | 0.8049 | 0.7805 | 0.8312 |
| bkl (165) | 0.7245 | 0.7461 | 0.7212 |
| df (17) | 0.7500 | 0.7778 | 0.8235 |
| mel (167) | 0.6667 | 0.6862 | 0.7006 |
| nv (1,007) | 0.9276 | 0.9321 | 0.9275 |
| vasc (21) | 0.9000 | 0.8261 | 0.9048 |

Minority precision/recall trade-offs explain why accuracy and macro-F1 disagree. Tiny df/vasc supports make their estimates unstable. Melanoma errors: 37→nv, 5→bkl; nv errors include 32→mel and 23→bkl. Retain class-wise evidence rather than choosing from accuracy alone.

## Curves and complementarity

Final train accuracy 95.96% versus val 86.36%; minimum val loss at epoch 11 (0.56464), increasing late loss despite improved accuracy. No LR reduction occurred because the same accuracy scheduler kept seeing improvements. Best at the epoch cap does not prove full convergence, but the generalization gap gives no guarantee that an unchanged extension will help. Logged optimizer updates 4,375/4,380 (five AMP skips), preserved as measured.

Aligned accuracy-winner predictions: **1,195 both correct; 98 only MobileNet correct; 103 only B0 correct; 107 both wrong; 228 prediction disagreements**. These are error diagnostics, not an ensemble result or a deployable oracle score. Complementarity justifies [one fixed 50/50 probability-fusion experiment](S04_PROPOSAL.md) before more GPU compute.

## Historical exploratory context, separate from controlled comparison

| Same exploratory cohort, different recipe/cost | Accuracy | Macro-F1 |
|---|---:|---:|
| S03 B0 single model | 86.36% | 0.7707 |
| Historical weighted B0+CBAM single model | 86.23% | 0.7808 |
| Historical strong-augmentation B0+CBAM | 85.30% | 0.7657 |
| Historical B0 four-stream ensemble | 88.16% | 0.8280 |
| Historical B0 + PanDerm eight-stream ensemble | 90.75% | 0.8573 |

Historical manifest hashes/class supports match, but heads, attention, batches, selection and ensemble costs differ. S03's +0.1331 accuracy points over weighted historical B0+CBAM is two images and accompanies lower macro-F1: **not an isolated CBAM effect**. Repeated historical validation selection may make ensemble maxima optimistic. Strict S01/historical strict tables remain separate; no new test result exists.

## Verified artifacts and preservation

- Run package: `results/structured_experiments/s03_efficientnet_b0_none_exploratory_seed42/`: history, config/environment/pretraining, primary/secondary metrics and all 1,503 predictions with seven probabilities, progress/log/summary/record, `closeout_verification.json`.
- `figures/`: seven PNG/PDF curves/matrices/class/support pairs, three CSVs. `figures_macro_f1/`: secondary matrix/class/support figures and CSVs. Accuracy curve and normalized matrix visually checked.
- `checkpoints/structured/s03_efficientnet_b0_none_exploratory_seed42/`: `best.pt`, `best_macro_f1.pt`, `latest.pt`; all epoch 20, identical selected/final model weights. CPU-loaded checkpoint/config/source hashes/history/predictions/metrics agree. Checkpoints/logs remain local; portable metrics/predictions/figures are committed.
- `results/model_comparison/structured/exploratory_screening_v1_seed42/`: controlled two-model table/graph. `s02_s03_matched/`: class deltas, paired errors and graph. `s03_exploratory_context/`: historical context table/graph.
- Master registry closed out; all other rows unchanged, including all 473 historical rows. No checkpoints retrained or replaced; original locked test unchanged.

Next: prepare fixed CPU fusion, then choose matched CBAM/augmentation or a justified alternative from that evidence. No automatic training queue. Future GPU launches stop after confirming log/checkpoint creation; user returns for completion analysis.

Current Phase Progress: 70%
Total Project Progress: 63%
