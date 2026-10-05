# S59–S61: supervised22k ConvNeXt-Tiny closeout

## S59 standalone result

S59 completed its authorized 20 epochs in **21.54 minutes** and stopped at the hard cap. The larger-pretraining hypothesis did not improve standalone performance. The successful frozen-feature gate was not a guarantee of better full fine-tuning.

| Selection | Epoch | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Best accuracy / final | 20 | 90.6853% | 0.850072 | 0.812359 | 0.828537 |
| Best macro-F1 | 18 | 90.6188% | 0.859338 | 0.814970 | 0.832974 |
| Original S06 ConvNeXt-Tiny accuracy winner | 17 | 91.7498% | — | — | 0.862831 |
| Original S18 ConvNeXt-Small accuracy winner | 17 | 92.2156% | — | — | 0.854948 |

S59's best accuracy is **1.0645 percentage points / 16 images below S06** and **1.5303 points / 23 images below S18**. S59 final weighted validation loss was 0.985582.

Class-wise scores for S59's accuracy-selected checkpoint:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| akiec | 0.755102 | 0.755102 | 0.755102 | 49 |
| bcc | 0.847222 | 0.792208 | 0.818792 | 77 |
| bkl | 0.838710 | 0.787879 | 0.812500 | 165 |
| df | 0.923077 | 0.705882 | 0.800000 | 17 |
| mel | 0.816456 | 0.772455 | 0.793846 | 167 |
| nv | 0.943853 | 0.968222 | 0.955882 | 1,007 |
| vasc | 0.826087 | 0.904762 | 0.863636 | 21 |

## Does epoch20 best justify more training?

| Epoch | Validation accuracy | Macro-F1 |
|---|---:|---:|
| 16 | 90.4857% | 0.817097 |
| 17 | 90.4192% | 0.826488 |
| 18 | 90.6188% | 0.832974 |
| 19 | 89.8869% | 0.818745 |
| 20 | 90.6853% | 0.828537 |

Epoch20 beats epoch18 by only **one correct image / 0.0665 percentage points**, while macro-F1 is lower. Training accuracy was 100% at epochs17–20 and training loss fell to 0.0000143. Validation accuracy oscillated; the data does not show a convincing continuing material gain. It is possible more epochs could change scores, but this evidence does not justify extending the run. No training extension was launched or configured.

## Error complementarity and two fixed ensemble tests

S59 fixes **24 of S53's 96 errors**, including 9 melanoma, 6 BKL, 6 nevus, 2 akiec and 1 BCC images. It also gets 68 images wrong that S53 gets right. Its standalone akiec recall (75.51%) exceeds S53 (71.43%), but most other class recalls are weaker.

Both fusion rules were declared before evaluation and used only S59's raw-accuracy winner at epoch20. No macro-F1 checkpoint, epochs, weights or thresholds were searched.

| Method | Accuracy | Macro-F1 | Melanoma recall | Gained / lost vs S53 |
|---|---:|---:|---:|---:|
| S53 reference equal five | 93.6128% | 0.886857 | 82.0359% | — |
| S60 add S59, equal six | 93.0805% | 0.882402 | 77.2455% | 7 / 15 |
| S61 replace S06 with S59, equal five | 93.0805% | 0.884610 | 79.0419% | 10 / 18 |

Both candidates lose **eight correct predictions / 0.5323 percentage points**; both fail the directional and material gates. Complementary hard errors did not translate into a useful probability average. Keep S53 as the provisional highest exploratory candidate (93.6128%); preserve GPU-reproduced S50 four-model reference (93.4797%). **Do not add S59 or replace the original Tiny in the ensemble.** Do not expand this into a weight or checkpoint search.

## Verification and saved artifacts

S59 best, macro-F1 best and latest checkpoints were hashed and strictly loaded into the exact timm architecture on CPU. All model states and latest optimizer state are finite. Epoch history and selected epochs agree with checkpoint metadata. Saved predictions independently reproduce accuracy, macro-F1 and confusion matrices; curves and raw/normalized confusion matrices, per-class CSVs and PNG/PDF figures exist for both selections. No training failure artifact exists; validation used FP32 throughout.

The history records six AMP/GradScaler skipped updates across 20 epochs; these were handled by the scaler. There was no recorded validation failure or invalid committed model/optimizer state. Selected checkpoint probability records were also checked directly against their exported prediction CSVs.

- Training results: `results/short_screening/supervised22k_transfer_v1/s59_convnext_tiny_supervised22k_exploratory_seed42/`
- Checkpoints: `checkpoints/short_screening/supervised22k_transfer_v1/s59_convnext_tiny_supervised22k_exploratory_seed42/`
- Checkpoint hashes/checks: training folder's `closeout_verification.json`.
- CPU fusion studies: `results/short_screening/s60_s61_s59_fusion/`, including predeclared plan, source/checkpoint manifests, metrics, probabilities/predictions, class comparisons, confusion matrices and PNG/PDF comparison figures.
- Master registry records S59 completed and both rejected fusion results.

No GPU inference/retraining or locked-test image/label access occurred during closeout. These are post-test exploratory development results with prior validation selection. The original frozen strict/test methodology and its **86.7598% original held-out test result** remain unchanged.
