# S01 closeout — strict EfficientNet-B0 control

Completed 3 October 2026: **20 epochs, epoch 18 selected by maximum validation macro-F1**. Closeout used saved artifacts and CPU only.

| Selected metric | Result |
|---|---:|
| Accuracy |85.62874251497006% |
| Macro precision |0.7742211258057381 |
| Macro recall / balanced accuracy |0.7852016013142199 |
| Macro-F1 |0.7791473336748238 |
| Weighted validation loss |0.6609002636758671 |
| Validation images |1,503 |
| Recorded training time |1,006.25s /16.77min |
| Parameters |4,016,515 |
| Peak allocated VRAM |851.87MiB; not total GPU reservation |

**Peak accuracy in history:**85.89487691284099% at epoch 15. That state was superseded by macro-F1 selection and is unavailable; do not combine its accuracy with epoch 18 F1 or call it a saved deployed checkpoint. Epoch20 ended at 85.63% accuracy/.7657 macro-F1 and is the resumable latest state.

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| akiec |50 |.7000 |.7000 |.7000 |
| bcc |77 |.7586 |.8571 |.8049 |
| bkl |165 |.7225 |.7576 |.7396 |
| df |18 |.8889 |.8889 |.8889 |
| mel |166 |.6121 |.6084 |.6103 |
| nv |1006 |.9374 |.9225 |.9299 |
| vasc |21 |.8000 |.7619 |.7805 |

Melanoma:101/166 correct, 38 predicted nv and 19 bkl;43 nv images were predicted mel. These confusions and weaker akiec/bkl performance remain improvement targets. Small df/vasc supports make their estimates uncertain.

Training accuracy reached93.36%, while later validation accuracy stayed near84–86%. Training loss fell; validation loss reached its minimum at epoch 11 then generally rose. This suggests a growing generalization gap, so another unchanged20-epoch extension is weakly motivated. Scheduler reductions precede epochs10 and 14; late macro-F1 improvement at 18 still matters.

## Fair historical context

| Same registered strict validation cohort | Selected epoch | Accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|---:|
| S01 B0 / common GAP-linear head |18 |85.63% |.7791 |60.84% |
| Historical weighted B0+CBAM / MLP head |16 |84.56% |.7569 |55.42% |

Deltas: **+1.0645 accuracy percentage points, +.022274 macro-F1, +5.4217 melanoma-recall percentage points**. Both use B0 ImageNet transfer, 224 RGB/flip and the same sqrt training-class weight formula. Historical architecture uses CBAM plus1280→512→128→7 BatchNorm/dropout head and batch 64; S01 uses GAP/dropout.2/linear7, microbatch 16/effective 32. Training order, dropout and loss aggregation also differ. This compares complete recipes; **it cannot isolate CBAM's effect or establish significance**. Validation loss aggregation differs, so loss values are not a direct fair comparison.

Comparison CSV, class deltas and PNG/PDF graph: `results/model_comparison/structured/s01_vs_legacy_b0/`. A matched new CBAM ablation is still required before an attention-effect claim.

## Verified package

`results/structured_experiments/s01_efficientnet_b0_none_strict_seed42/` contains config/environment/pretraining records, 20-epoch history, selected metrics, progress/training summary, local train.log and all 1,503 image IDs/true labels/predictions/seven-class probabilities. Predictions recompute the saved matrix and metrics. All seven PNG/PDF figure pairs and three CSVs passed checks; accuracy curve and normalized matrix were visually inspected. See `closeout_verification.json` for checks/hashes.

Local `checkpoints/structured/s01_efficientnet_b0_none_strict_seed42/{best.pt, latest.pt}` were loaded on CPU: selected weights/metrics agree with the best state in latest. Best epoch 18; latest epoch 20. Checkpoints/logs stay local; metrics/predictions/figures are portable Git evidence.

Master registry status is completed, decision `strict_reference_retained_screening_moves_to_exploratory`; all 473 historical rows preserved. Test images were never opened. Next: [two-stage strategy](TWO_STAGE_STRATEGY.md) and [S02 proposal](S02_PROPOSAL.md).
