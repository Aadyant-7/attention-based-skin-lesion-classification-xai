# S79/S80 completed closeout — 8 October 2026

Completed and independently verified from saved checkpoints/predictions. No additional training, inference, alternative ensemble or original-test evaluation was performed during verification. Stop for tonight; further work requires the next user instruction.

## Results: same 1,503-image exploratory validation cohort

| Saved result | Epoch | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| S79 ConvNeXt-Tiny + CBAM, accuracy-selected | 33 | 93.013972% | 0.885548 | 0.870631 | 0.877434 |
| S79 macro-F1-selected | 35 | 92.481703% | 0.900174 | 0.858501 | 0.877708 |
| S79 final state | 40 | 92.614770% | 0.876091 | 0.871125 | 0.873314 |
| S80 fixed equal-five CBAM replacement | 33 candidate | 93.346640% | 0.913249 | 0.849629 | 0.877017 |
| Retained S53 equal-five reference | Fixed historical selections | 93.612774% | See reference metrics | See reference metrics | 0.886857 |

S79's accuracy-selected checkpoint got 1,398 correct /105 incorrect. Its melanoma recall was 131/167 (78.4431%); S80 melanoma recall was 132/167 (79.0419%), below S53's 137/167 (82.0359%). Complete seven-class scores and confusion matrices are saved for each selection and final state.

S79 exceeds the earlier S06 ConvNeXt-Tiny accuracy result (91.7498%) by about **1.264 percentage points**. The new package changes attention, augmentation, training duration and checkpoint-selection policy; this does not isolate a causal CBAM improvement. S53's Tiny component uses its macro-F1-selected checkpoint, while S79 uses its predeclared accuracy winner.

S80 replaces only Tiny with S79, retaining ConvNeXt-Small, DenseNet201, EfficientNetV2-S and EfficientNet-B0 at equal weights 0.2. It gains 11 reference errors but loses 15 reference successes: **four fewer correct predictions**, -0.266134 percentage points accuracy and -0.009841 macro-F1. It fails the material-improvement gate. Preserve the attention-inclusive result honestly; retain S53 as the stronger single-image exploratory reference. No rescue weight/member/epoch search was run.

## Did the curve justify more epochs?

| Epoch | Validation accuracy | Macro-F1 |
|---:|---:|---:|
| 33 | 93.0140% | 0.877434 |
| 34 | 92.5482% | 0.871981 |
| 35 | 92.4817% | 0.877708 |
| 36 | 92.0160% | 0.866360 |
| 37 | 92.3486% | 0.862500 |
| 38 | 92.4817% | 0.866310 |
| 39 | 91.6833% | 0.868305 |
| 40 | 92.6148% | 0.873314 |

The run stopped at its authorized **40-epoch cap**, not early stopping. Longer training beyond 20 genuinely helped: accuracy rose from 91.3506% at epoch20 to 93.0140% at epoch33. However, epochs34–40 never exceeded that accuracy or made another meaningful improvement. The small F1 record at35 did not meet the meaningful-gain threshold. LR reductions occurred at25 and37; the final three epochs after the second reduction remained below the best accuracy. Training accuracy reached 99.8716%, while validation fluctuated: evidence of a generalization plateau rather than a sustained upward trend. Further improvement is possible, but this curve does not establish that extending past40 would materially help. No extension was launched.

## Numerical recovery and validity

The original failed partial epoch21 was replayed from the unchanged, preserved epoch20 checkpoint. Standard GradScaler skip/backoff handling was the only recovery amendment; architecture/configuration, split, loss, optimizer recipe and FP32 validation remained frozen. The resumed run recorded three nonfinite-gradient events at epoch21/batch117, epoch22/batch174 and epoch32/batch80, all identifying `features.1.0.block.0.bias`. These optimizer updates were skipped safely with scaler backoff; no FP32 training fallback was used. Model/optimizer state and all validation probabilities remained finite. Successful optimizer updates: 8,757 out of 8,760 attempted effective batches.

Verification passed: full-state checkpoint at40; earliest raw-best accuracy/F1 checkpoint epochs33/35 and model tensors; contiguous 40-epoch history; metrics recomputed from saved predictions; class order/cohort/probability sums; confusion matrices and class scores; PNG/PDF figures; unchanged frozen sources/configuration; preserved epoch20 backup and failure evidence. All 569 prior registry rows are unchanged, and S79/S80 are registered as completed. Verification initialized no GPU and loaded no test labels/images.

## Saved artifacts (paths relative to repository root)

- `results/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/`: selected accuracy metrics/predictions/probabilities, `history.csv`, `train.log`, summaries, recovery evidence, `verification.json`, class/confusion/training PNG/PDF figures.
- Its `macro_f1_selected/`, `final_epoch/` and `fixed_ensemble/` folders hold the three other complete metric/prediction/class/confusion packages. Ensemble comparison figures are in `fixed_ensemble/comparison_figures/`.
- `checkpoints/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/`: `best.pt` (33), `best_macro_f1.pt` (35), `latest.pt` (40), immutable epoch20 backup and diagnostic snapshot. Large checkpoint binaries remain local, excluded from Git.
- `results/master_experiment_registry.csv`: completed S79 and S80 rows.

These are exploratory validation results on a repeatedly used image-level development split, not independent test estimates. The original single locked-test result remains 86.7598% /0.794473 macro-F1 and was not reevaluated.
