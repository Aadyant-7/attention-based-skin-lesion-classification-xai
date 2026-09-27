# Strict baseline snapshot before accuracy exploration

Frozen on 2026-09-28 from Git commit `a31f956bde8d5bdc1ccd08af7cb219165606a9c3` on `main`. This document records the lesion-disjoint, image-only benchmark and all completed validation analyses before any exploratory methodology is introduced. Source CSV/JSON files retain the full precision and per-class results; all numbers below are validation results. The strict test partition has no evaluation artifacts or reported metrics in this repository, and the training and analysis scripts use train/validation only. Its non-use outside this repository cannot be independently proven.

## Data, environment, and split

- Dataset: `data/raw/HAM10000/`, with `HAM10000_metadata.csv` and `HAM10000_images_part_1/` plus `HAM10000_images_part_2/`; 10,015 images and 7,470 lesion IDs. Raw images and checkpoints are local and Git-ignored.
- Original diagnosis counts: akiec 327, bcc 514, bkl 1,099, df 115, mel 1,113, nv 6,705, vasc 142.
- Permanent split: `data/splits/split_assignments.csv`, generated with `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)` and fold groups 0–2 test, 3–5 validation, remainder training. Train 7,009 images/5,230 lesions; validation 1,503/1,122; test 1,503/1,118. Pairwise lesion-ID overlap: **zero** for train/validation, train/test, and validation/test.

| Partition | akiec | bcc | bkl | df | mel | nv | vasc | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 228 | 359 | 769 | 80 | 779 | 4,694 | 100 | 7,009 |
| Validation | 50 | 77 | 165 | 18 | 166 | 1,006 | 21 | 1,503 |
| Test, locked | 49 | 78 | 165 | 17 | 168 | 1,005 | 21 | 1,503 |

- Local environment at snapshot: Windows; Python 3.14.4; PyTorch 2.11.0+cu128; torchvision 0.26.0+cu128; CUDA build 12.8; RTX 4060 8,188 MiB, NVIDIA driver 591.86; NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, Pillow 12.3.0. See `requirements.txt` for declared dependencies.

## Training method

`src/models.py` uses torchvision EfficientNet-B0 with `EfficientNet_B0_Weights.DEFAULT` ImageNet weights. The feature extractor emits 1,280 channels; one CBAM block is placed **after the entire backbone and before adaptive average pooling**. `src/cbam.py` applies channel attention (shared two-layer 1×1 MLP, reduction 16, average/max global maps), then spatial attention (channel average/max maps, 7×7 convolution). The head is `AdaptiveAvgPool2d(1) → Flatten → Dropout(0.3) → Linear(1280,512) → BatchNorm → ReLU → Dropout(0.2) → Linear(512,128) → BatchNorm → ReLU → Dropout(0.1) → Linear(128,7)`.

`src/data.py` resizes the original 600×450 RGB image to **224×224** (changes aspect ratio), then ToTensor and ImageNet normalization (`[.485,.456,.406]` / `[.229,.224,.225]`). Training alone adds horizontal flip with probability .5. No lesion crop, hair removal, color constancy, CLAHE, vertical flip, rotation, or color jitter is used. Batch size 64; two workers; AMP. AdamW uses backbone LR 3e-5, CBAM/head LR 1e-4, weight decay 1e-4. `ReduceLROnPlateau` tracks validation macro F1 (factor .5, patience 2). Runs allow 20 epochs and stop after five stale epochs. The best checkpoint and `validation_metrics.json` are selected by validation macro F1, with improvement threshold 1e-5. Seed 42. Weighted CE and focal alpha use train-only `sqrt(N/(K*n_c))`, normalized to mean one; focal gamma is 2. Full random oversampling repeats training metadata rows to 4,694 per class (32,858 samples/epoch) and uses ordinary CE.

## Completed experiments

`—` in best epoch means a validation-only inference or fitted-feature analysis, with no new CNN checkpoint. The source files below contain all candidate grid rows, not just the selected rows.

| Experiment / protocol | Validation accuracy | Macro F1 | Best epoch | Main conclusion |
|---|---:|---:|---:|---|
| Weighted CE, normal | 0.8456420492348636 | 0.7568732673677362 | 16 | Best original trained checkpoint for balanced metrics. |
| Random oversampling, normal | 0.852960745176314 | 0.732131726806851 | 17 | Better accuracy, worse macro F1; repeated-image overfit. |
| Focal loss, normal | 0.8376580172987359 | 0.7238217217065802 | 16 | Did not improve either leader. |
| Weighted CE, horizontal-flip TTA | 0.8423153692614771 | 0.7703611681312049 | — | Macro F1 improves, accuracy falls. |
| Oversampled, horizontal-flip TTA | 0.8576180971390552 | 0.7296021576799175 | — | Pre-Phase-5C accuracy leader, weak macro F1. |
| Weighted TTA .9 + focal normal .1 | 0.8436460412508316 | **0.7718103328854748** | — | Overall macro-F1 leader; tuned on validation. |
| Weighted TTA .2 + oversampled TTA .8 | 0.8576180971390552 | 0.7398508297313039 | — | Ensemble matches, but does not exceed, pre-Phase-5C accuracy leader. |
| Frozen post-CBAM embeddings + logistic regression C=.3 | 0.8383233532934131 | 0.7347224551656346 | — | Frozen-feature ML falls behind CNN. |
| Weighted TTA .9 + frozen-feature LR .1 | 0.8449767132401863 | 0.7712452038039167 | — | Fusion misses macro-F1 leader. |
| Metadata-only classifier | 0.278776 | 0.185897 | — | Age/sex/localization alone are weak. |
| Image embeddings + metadata small MLP | 0.8423153692614771 | 0.7454918512517936 | — | Best standalone multimodal classifier does not justify CNN retraining. |
| Weighted TTA .9 + multimodal MLP .1 | 0.8456420492348636 | 0.7709512216044716 | — | Best metadata-aware macro F1, below image-only leader. |
| Weighted TTA .8 + multimodal MLP .2 | 0.846307385229541 | 0.7637399804202419 | — | Best metadata-aware accuracy, below image-only leader. |
| MC dropout, 30 passes + flip | 0.8416500332667998 | 0.7672831786435367 | — | Best MC setting misses leaders; see 10/20/30-pass grid. |
| Aspect-preserving resize + center crop diagnostic | 0.8163672654690619 | 0.7158782015978372 | — | Inference-only field-of-view change hurts current checkpoint; does not isolate geometry. |
| Oversampled TTA, train-prior correction λ=.2 | **0.8589487691284099** | 0.7227228069686288 | — | Current accuracy leader by 0.001331; majority-class shift reduces macro F1. |
| Weighted TTA, cost-weight correction λ=.1 | 0.8436460412508316 | 0.7661751929662156 | — | Best corrected macro F1, below uncorrected ensemble. |

The original requested 0.857618 accuracy reference predates Phase 5C. The **current highest accuracy is 0.8589487691284099** from validation-tuned prior correction; **current highest macro F1 remains 0.7718103328854748** from the weighted/focal ensemble. These are maxima over different methods, not one model's paired metrics. Grid selection on this same validation set can be optimistic; neither figure is a test estimate. Melanoma/nevus and benign keratosis/nevus confusion remain material, and nevus dominates the dataset.

## Source artifacts and failures

- Training ledger: `results/experiments.csv`; configs `configs/first_run.json`, `configs/oversampled_v1.json`, `configs/focal_v1.json`; runs `results/runs/<run_name>/{history.csv,validation_metrics.json,config.json}`; checkpoints `checkpoints/<run_name>/{best.pt,latest.pt}` for each of `efficientnet_b0_cbam_weighted_v1`, `efficientnet_b0_cbam_oversampled_v1`, and `efficientnet_b0_cbam_focal_v1`. Checkpoints remain local and ignored; the weighted run's post-training CSV `KeyError` was repaired without retraining.
- Normal/flip TTA: `results/validation_inference_comparison.csv` and `results/validation_inference/`.
- CNN ensembles, frozen-feature classifier grid, feature/CNN fusion, confusion analysis: `results/phase4/` and `results/validation_error_analysis.md`. Ensembles bring small macro improvement; frozen ML does not improve the leader.
- Metadata feasibility, field ablations, classifier/fusion grids: `results/phase5_multimodal_probe/`. No metadata setting cleared the prespecified +0.015 improvement gate; full multimodal training was stopped.
- MC dropout grid, aspect-ratio diagnostic, and preprocessing audit: `results/phase5b_inference_audit/`. All 7,009 training images measured 600×450; an 8/150 sampled border heuristic flagged obvious dark borders. MC dropout did not improve the leader.
- Prior/logit correction and optional combination grid: `results/phase5c_prior_correction/`. The accuracy increase predicts more nevus and correctly classifies fewer melanoma images; full-strength correction degrades macro F1. No training or test access occurred in that analysis.

Strict methods and their source artifacts remain on `main`. Any later image-level or otherwise weaker protocol must be named as exploratory and must write to separate split/result/checkpoint paths. The locked strict test set remains reserved for a final evaluation after methodology selection.
