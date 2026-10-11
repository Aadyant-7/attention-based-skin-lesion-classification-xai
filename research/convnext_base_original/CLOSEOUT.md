# R203 completed; training closed

11 October 2026: ConvNeXt-V2 Base completed all **30 epochs** on the original exploratory split. Checkpoints, histories and previous research remain preserved. Two fixed CPU-only fusions were completed afterward; no further training, model inference or test evaluation was performed.

## Verified standalone results

| Selection | Branch | Epoch | Validation accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Accuracy-selected `best.pt` | Raw | 20 | **93.213573%** | 0.888528 |
| Macro-F1-selected `best_macro_f1.pt` | EMA | 27 | 93.013972% | **0.890102** |
| Final raw | Raw | 30 | 93.213573% | 0.886587 |
| Final EMA | EMA | 30 | 93.013972% | 0.886720 |

EMA's accuracy winner at epoch 25 ties raw's 93.213573%; the predeclared earliest-epoch rule correctly retains raw epoch 20. The accuracy and macro-F1 maxima belong to different checkpoints.

Accuracy-selected checkpoint: **1,401 correct / 102 incorrect** of 1,503; macro precision **0.913368**, macro recall **0.868953**, weighted-F1 **0.931338**, unweighted validation CE **0.488037**.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| akiec | 0.8200 | 0.8367 | 0.8283 | 49 |
| bcc | 0.9114 | 0.9351 | 0.9231 | 77 |
| bkl | 0.8896 | 0.8788 | 0.8841 | 165 |
| df | 1.0000 | 0.7647 | 0.8667 | 17 |
| mel | 0.8684 | 0.7904 | 0.8276 | 167 |
| nv | 0.9542 | 0.9722 | 0.9631 | 1,007 |
| vasc | 0.9500 | 0.9048 | 0.9268 | 21 |

Main remaining confusions: melanoma→nevus 30, nevus→melanoma 15, bkl→nevus 12, nevus→bkl 10. Relative to S83, Base improves akiec recall (41/49 versus 35/49), but melanoma recall is lower (132/167 versus 137/167).

Base exceeds S06 ConvNeXt-Tiny's 91.749834% accuracy by **1.463739 percentage points** and its accuracy-selected macro-F1 by **0.025697**. It exceeds S79's 93.013972% accuracy by three images, with better macro-F1. This is a comparison of complete training packages: V2 architecture, model capacity, pretraining, augmentation, optimizer/schedule and training duration differ; it does not isolate a capacity or EMA causal effect. R202's different 902-image V2 result remains separate.

## Final bounded CPU ensemble comparison

Before computing fusion metrics, `ensemble_comparison/FIXED_PLAN.json` recorded exactly two combinations. Both use only R203's raw epoch-20 accuracy-selected probabilities; no other epochs, EMA branches, weights, calibration or combinations were searched.

| Method | Correct | Accuracy | Macro-F1 | Gained / lost vs S83 | Melanoma recall |
|---|---:|---:|---:|---:|---:|
| Existing S83 equal-six | 1,413 | **94.011976%** | 0.895201 | — | 82.0359% |
| R204: add Base, equal-seven | 1,412 | 93.945442% | 0.891350 | 11 / 12 | 80.8383% |
| R205: replace Small with Base, equal-six | 1,410 | 93.812375% | **0.896011** | 11 / 14 | 80.8383% |

Addition uses equal 1/7 per constituent, not a 50/50 mixture of the old ensemble and Base. Replacement removes only S18 ConvNeXt-Small and retains equal 1/6 per constituent. The same ordered 1,503 validation images, class labels and lesions are verified. Original S83 probabilities are reproduced from the exact five saved FP32 member caches plus S79's saved macro-F1-selected probabilities.

Base alone fixes **42 S83 errors**; S83 fixes **54 Base errors**; **48 errors are shared**. These counts describe error complementarity, not an executable oracle classifier. Actual soft voting fixes 11 errors but introduces 12 or 14 errors. Both candidates fail the existing material-improvement gate (at least eight net correct, +0.005 accuracy, nondecreasing macro-F1 and melanoma recall). R205's +0.000810 macro-F1 is a small trade-off with three fewer correct images, not an accuracy improvement. **Retain S83; do not add Base to the final accuracy-focused ensemble on this evidence.**

Retained S83 components and existing selectors: ConvNeXt-Tiny (macro-F1), ConvNeXt-Small (macro-F1), DenseNet201 (macro-F1), EfficientNetV2-S (macro-F1), EfficientNet-B0 (accuracy), ConvNeXt-Tiny+CBAM (macro-F1). Equal 1/6 FP32 identity probability fusion. CBAM remains in the S79 component; Base has native GRN and no added CBAM.

## More epochs?

**Do not extend R203.** Accuracy first peaked at epoch 20 and was never exceeded in the remaining ten epochs; epoch 30 merely ties it. EMA macro-F1 improved genuinely at 27, then decreased for the last three epochs. Training accuracy reaches **99.93%**; raw validation loss rises from **0.4880 at 20 to 0.5590 at 30**, and EMA loss from **0.3718 to 0.5313**. The cosine schedule is finished at stage-3 LR **1e-7**. This suggests a late plateau and increasing overconfidence, not a demonstrated rising accuracy trajectory. More epochs at that LR offer little evidence-based benefit; restarting the schedule would be a different experiment.

**GPU training is now closed by the user's direction.** No extension, backbone, ensemble search or test run is queued. Next work is drafting and arranging the existing verified result tables/figures and methodology/XAI evidence; report-format instructions will guide that writing.

## Verification, provenance and locations

- All **eight checkpoint hashes**, finite weights/optimizer/EMA state, launch config/code/split/pretraining/runtime and raw/EMA selections verified on CPU.
- Thirty contiguous epochs; 7,009 natural originals accepted each epoch; **6,600 optimizer updates / 6,160 EMA updates**; no numerical failures or skipped updates. All group LRs match the frozen schedule. Runtime **5,522.88 seconds (~92 minutes)**; peak allocated VRAM **3.65 GiB**.
- All **nine prediction packages** verified against the cohort, metrics, per-class CSVs, raw/normalized confusion matrices and probability CSVs. **40 PNGs decoded and 40 PDF headers/end markers checked**. Training/EMA/LR curves and branch comparison figures exist.
- New fusion packages contain metrics, probabilities/predictions, class scores, raw/normalized confusion matrices, PNG/PDF figures, changed predictions, source hashes, complementarity, comparison table/figure and local/master registry entries. Independent recomputation agrees with both fusion results.
- R203 results: `results/structured_experiments/convnext_base_original_v1/r203_convnextv2_base_original_exploratory_seed42/`.
- R203 local checkpoints: corresponding `checkpoints/structured_experiments/convnext_base_original_v1/r203_convnextv2_base_original_exploratory_seed42/`.
- Audit receipts: `verification.json` (checkpoints/LR), `artifact_verification.json` (predictions/figures).
- Fusion results: `results/structured_experiments/convnext_base_original_v1/ensemble_comparison/` (`r204_equal_seven/`, `r205_replace_small_equal_six/`, comparison figures, fixed plan and verification).
- Reproducible CPU closeout/fusion verification: `python -B -m research.convnext_base_original.closeout`. Completed train logs are also copied into `train_log.txt` for portable evidence; original log is preserved.

These scores are **exploratory validation**, not test accuracy. The existing development split shares 563 train/validation lesions, affecting 596 validation images, and the validation set has been reused extensively. Historical test outcomes are already known. No test labels, probabilities or images were used for these fusions, and no test evaluation occurred in this closeout. Historical strict results and historical S83 test audit remain unchanged and must be reported separately. Do not promise that 94.01% validation transfers to test or clinical performance.
