# S06 closeout — ConvNeXt-Tiny exploratory screening

**Completed, verified on CPU; no new GPU run or test-image inference.** ID `s06_convnext_tiny_none_exploratory_seed42`.

## Saved states

| State | Epoch | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted loss |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy winner | 17 | 91.7498% | 0.880435 | 0.853326 | 0.862831 | 0.529672 |
| Macro-F1 winner | 18 | 91.4837% | 0.873533 | 0.866726 | 0.868746 | 0.489416 |
| Final/latest | 20 | 90.7518% | 0.866553 | 0.843270 | 0.853366 | 0.545264 |

Do not combine accuracy from epoch17 with F1/classes from epoch18. `best.pt` selects accuracy; `best_macro_f1.pt` selects F1; `latest.pt` is full epoch20 recovery state.

Runtime **14.94 minutes**, 27,825,511 parameters, peak allocated VRAM **1503.34MiB** (not total reservation). 4,375/4,380 optimizer updates: five AMP skips. Final train accuracy **99.71%** vs final val **90.75%**; minimum weighted validation loss at epoch5. Later accuracy improved despite rising loss, consistent with overconfident errors/generalization gap; do not simply extend epochs or claim calibration.

## Fair exploratory comparison

| Method | Accuracy | Macro-F1 | Model passes |
|---|---:|---:|---:|
| S02 MobileNetV3-Large | 86.0279% | 0.786111 | 1 |
| S03 EfficientNet-B0 | 86.3606% | 0.770660 | 1 |
| S05 B0 + CBAM | 86.3606% | 0.769301 | 1 |
| S04 fixed S02/S03 fusion | 88.5562% | 0.805654 | 2 |
| S06 ConvNeXt-Tiny | 91.7498% | 0.862831 | 1 |

ConvNeXt improves accuracy **5.39 percentage points** over S03/S05 and **3.19 points** over S04, with 81 and 48 additional correct images respectively. Accuracy-selected ConvNeXt correctly classifies **1,379/1,503**. This is common-budget model-package evidence, not an isolated topology effect, statistical significance or lesion-independent/test performance. External pretraining/native architecture differ; S04 costs two model passes.

| Class (support) | Precision | Recall | F1 |
|---|---:|---:|---:|
| akiec (49) | 0.8919 | 0.6735 | 0.7674 |
| bcc (77) | 0.7789 | 0.9610 | 0.8605 |
| bkl (165) | 0.9122 | 0.8182 | 0.8626 |
| df (17) | 0.8750 | 0.8235 | 0.8485 |
| mel (167) | 0.8025 | 0.7784 | 0.7903 |
| nv (1007) | 0.9502 | 0.9662 | 0.9581 |
| vasc (21) | 0.9524 | 0.9524 | 0.9524 |

Versus S03, every class F1 improves; melanoma recall **70.06%→77.84%**, melanoma F1 **.6862→.7903**, bkl F1 **.7461→.8626**. Important residual weakness: akiec recall33/49 (**67.35%**); its ten bcc confusions are the largest akiec error group. Minority supports df17/vasc21 make their scores sensitive to a few examples.

## Complementarity and bounded follow-up

Of 124 S06 errors, S02 recovers46 and S03 recovers52; S05 recovers48 and S04 recovers49. S06 in turn fixes132 S02 errors and133 S03 errors. Complementarity is real in the aligned validation predictions, but correctness-dependent oracle unions are diagnostic upper bounds, not evaluated fusion performance.

The three candidates previously specified in the strategy were frozen before new fusion scoring: equal S02+S06 (S07), S03+S06 (S08), and S02+S03+S06 (S09). All completed on CPU. None increases accuracy over S06; S09 raises macro-F1 to.8691 with three-model cost. See [bounded ensemble closeout](S07_S09_CLOSEOUT.md). Preserve S06 as the simple accuracy/cost reference and S09 as a balanced alternative; recommend [one stronger complementary backbone](S10_PROPOSAL.md) before more fusion.

## Verification and artifacts

- `results/structured_experiments/s06_convnext_tiny_none_exploratory_seed42/`: both metrics/prediction sets, 20-epoch history, config/environment/pretraining/progress/summary/record, `closeout_verification.json` and local `train.log`.
- `figures/`: seven PNG/PDF curve/matrix/class/support pairs and three CSVs; `figures_macro_f1/`: four secondary matrix/class/support pairs and three CSVs.
- `checkpoints/structured/s06_convnext_tiny_none_exploratory_seed42/`: own trusted best/F1/latest CPU-loaded; config, launch source, history, model state, selected probabilities and record hashes authenticated.
- `results/model_comparison/structured/s06_backbone_review/`: five-method comparison, class table, aligned error CSV, overlap JSON; common training table updated separately.
- Checkpoint/metric/prediction hashes in verification JSON; master registry closed out. All other rows unchanged including473 historical rows. Curves/comparison/matrix visually checked.

Exploratory manifest unchanged (SHA `75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb`),7,009/1,503/1,503 (~70/15/15),563 shared train/val lesions. Original test unchanged and excluded. Fresh strict confirmation comes later; do not evaluate exploratory checkpoints on contaminated strict validation.

Current Phase Progress: 86%
Total Project Progress: 72%
