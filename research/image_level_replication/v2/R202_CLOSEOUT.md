# R202 ConvNeXt-V2 Base — completed closeout

Completed 11 October 2026 (Asia/Calcutta): all **50 planned epochs** committed. No new experiment, ensemble, outer-assessment inference or test evaluation was performed during closeout.

## Actual results

| Selection | Branch | Epoch | Inner-validation accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Accuracy-selected `best.pt` | EMA | 20 | **93.2373%** | 0.875277 |
| F1-selected `best_macro_f1.pt` | Raw | 40 | 93.1264% | **0.888584** |
| Final raw weights | Raw | 50 | 92.5721% | 0.880502 |
| Final EMA weights | EMA | 50 | 92.4612% | 0.879567 |

Accuracy-selected metrics: accuracy `0.9323725055432373`, macro precision `0.8699997508244776`, macro recall `0.8849864193457628`, macro-F1 `0.8752770626299834`, weighted-F1 `0.9325150783293663`, unweighted validation CE `0.42569089835158697`. **841 correct / 61 incorrect** out of 902.

Raw accuracy and F1 both peaked at epoch 40. EMA macro-F1 peaked at epoch 25 (`0.8873800841005618`, accuracy 93.1264%). Raw/EMA selection and earliest-epoch tie rules remain unchanged. Accuracy and maximum macro-F1 come from different checkpoints and must be reported separately.

### Class results at the accuracy-selected checkpoint

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| akiec | 0.8276 | 0.8000 | 0.8136 | 30 |
| bcc | 0.8936 | 0.9130 | 0.9032 | 46 |
| bkl | 0.8763 | 0.8586 | 0.8673 | 99 |
| df | 0.6923 | 0.9000 | 0.7826 | 10 |
| mel | 0.8300 | 0.8300 | 0.8300 | 100 |
| nv | 0.9702 | 0.9702 | 0.9702 | 604 |
| vasc | 1.0000 | 0.9231 | 0.9600 | 13 |

Melanoma recall is 83/100; akiec recall is 24/30. Class scores and raw/normalized confusion matrices are saved for both selected checkpoints and both final branches; minority-class support remains small.

## Interpretation and convergence

On the same frozen V2 inner cohort, R201's stopped B3 package achieved 85.8093% accuracy / 0.757946 macro-F1 at its accuracy-selected checkpoint. R202 improves these by **7.4279 percentage points** and **0.117331 macro-F1**. This is a descriptive comparison of complete training packages: R201 stopped at epoch 28 and used a different recipe and repeated balanced exposures. It does not isolate an architecture-only effect.

The accuracy peak was already reached at epoch 20; epochs 21–50 did not surpass it. Later training improved raw macro-F1 through epoch 40, so the late phase was useful for class balance, but it did not establish a higher accuracy. The meaningful-improvement ledger last advanced at epoch 25; numerical best-F1 improvements below its threshold are still correctly preserved. Stopping reason is the predeclared 50-epoch maximum.

This is **single-fold, post-development image-level inner validation with natural lesion overlap**, using 8,111 training / 902 validation originals. The 1,002-image V2 outer assessment was excluded. It is neither test accuracy, strict lesion-independent performance nor completed ten-fold cross-validation. Historical S83's 94.011976% validation uses the older 1,503-image exploratory cohort; retain it separately. Historical models cannot fuse directly with R202 on V2 because their training includes V2 assessment images.

## Recovery and provenance

Two checkpoint-serialization failures occurred after epochs 16 and 20 during disk-space pressure. Atomic `latest.pt` retained committed epochs 15 and 19 respectively; recovery replayed only the uncommitted epochs. Failure JSONs and diagnostic checkpoints remain preserved. The pending drive-move stop request was archived when continuation was authorized. No project relocation or training-recipe change occurred.

No nonfinite training/validation events were recorded. Completed counters are **12,700 accepted optimizer updates / 12,192 EMA updates**. Fifty contiguous epoch records, RNG, optimizer and LR state remain in `latest.pt`; final raw and EMA weights remain in `final.pt`. Full history, raw/EMA curves, LR history/curves, predictions/probabilities, class scores, confusion matrices and PNG/PDF figures are saved. Both experiment registries record `completed`, 50 epochs and the EMA epoch-20 accuracy selection.

CPU-only closeout verification passed: eight checkpoint hashes and selected/final state consistency, all nine saved prediction packages on the identical 902-image cohort, recomputed metrics/class scores/matrices, 40 decoded PNGs and 40 structurally valid PDFs. Verification is recorded in `verification.json`. The historical preservation audit also passed (636 protected files, 588 historical registry rows, 127 checkpoint inventory entries); see `historical_preservation_verification.json`. `train_log.txt` is a portable copy of the completed local `train.log`, including interrupted attempts and recoveries.

## Locations

- Results: `results/image_level_replication/v2/r202_convnextv2_base_fold00_seed42/`.
- Checkpoints (local, Git-ignored): `checkpoints/image_level_replication/v2/r202_convnextv2_base_fold00_seed42/`.
- Accuracy package: `best_accuracy/`; F1 package: `best_macro_f1/` inside the result folder.
- Branch packages: `raw_accuracy/`, `raw_macro_f1/`, `ema_accuracy/`, `ema_macro_f1/`, `final_raw/`, `final_ema/`.
- Curves: `figures/` and `figures/ema/`; raw/EMA comparison: `comparison_figures/`.
- Exact metrics/selector records/checkpoint hashes: `closeout.json`; configuration: `config.json`; provenance: `launch_manifest.json` and `environment.json`.
- Registries: `results/master_experiment_registry.csv` and `results/image_level_replication/v2/experiment_registry.csv`.

The complete prepared recipe and commands remain in [R202_PHASE1_PROPOSAL.md](R202_PHASE1_PROPOSAL.md). Work stops after this closeout; future decisions remain for the next session.
