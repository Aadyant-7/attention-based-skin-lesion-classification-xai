# Attention-Based Skin Lesion Classification with Explainable AI

**Status:** Phase 1 complete; validation experiments continue on the separate `accuracy-exploration` branch. No locked test-set result is reported.

This project classifies seven HAM10000 skin lesion categories from **images only**. It fine-tunes an ImageNet-pretrained EfficientNet-B0, applies a Convolutional Block Attention Module (CBAM) to the final spatial feature map, then uses global pooling and a seven-logit classifier. CBAM learns channel and spatial emphasis during prediction. Grad-CAM is a later explanation of a prediction; it does not improve accuracy.

## Dataset and leakage policy

Obtain HAM10000 from its authorized distribution and place the original `HAM10000_metadata.csv`, `HAM10000_images_part_1/`, and `HAM10000_images_part_2/` under `data/raw/HAM10000/`. The raw images are excluded from Git because of their size and dataset licensing. Metadata supplies labels, image IDs, and `lesion_id` for grouping; age, sex, and localization are not model inputs.

Run `python -m scripts.prepare_data` once. This writes `data/splits/split_assignments.csv` and `label_mapping.json`. The fixed seed 42 split uses `StratifiedGroupKFold` with 20 folds: folds 0–2 are test, 3–5 validation, and 6–19 training. Every image of a lesion stays in one partition. The saved split is reused, checked against the raw metadata, and never regenerated silently. Training and model selection use only train and validation. Do not inspect test metrics until the configuration is locked.

HAM10000 is imbalanced. The first experiment uses training-frequency-only class-weighted cross entropy, with weight `sqrt(N / (K * n_c))` normalized to mean 1. This moderates minority emphasis while retaining natural training sample frequency. The controlled second experiment repeats training image references with seed 42 until each class matches the majority count and uses ordinary unweighted cross entropy. Repeated references receive fresh random training augmentation when loaded; no image files are copied. Validation and test retain their original distributions.

## Setup and run

On Windows, create or activate `.venv`, install a CUDA-capable PyTorch and torchvision pair appropriate for the machine from the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install `requirements.txt`. The current project environment uses PyTorch 2.11.0 and torchvision 0.26.0 with CUDA 12.8. Check `python -c "import torch; print(torch.cuda.is_available())"` before training.

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data
.\.venv\Scripts\python.exe -m scripts.sanity
.\.venv\Scripts\python.exe -m src.train --config configs/first_run.json
.\.venv\Scripts\python.exe -m scripts.preflight_oversampling
.\.venv\Scripts\python.exe -m src.train --config configs/oversampled_v1.json
```

If interrupted, run the same training command or add `--resume`. A compatible `latest.pt` is automatically resumed; `--resume` requires it. Best checkpoint selection maximizes validation macro F1. ReduceLROnPlateau responds to validation macro F1 and early stopping uses five non-improving epochs. The first run is limited to 20 epochs. AdamW uses a `3e-5` backbone learning rate, `1e-4` for CBAM/head, and `1e-4` weight decay. Batch size 64 was checked on the RTX 4060 (about 2.9 GiB peak allocated in a synthetic AMP optimizer step). Input is 224×224 with ImageNet normalization; only horizontal flip at `p=0.5` augments training. Validation is deterministic.

After configuration lock in a later phase, explicit test evaluation is available with `python -m src.evaluate --checkpoint <best.pt> --output <metrics.json> --confirm-locked-test`. Grad-CAM examples can be produced with `python -m src.gradcam --checkpoint <best.pt> --image <image.jpg> --output <overlay.png>`.

## Repository layout

- `src/`: image pipeline, EfficientNet-CBAM, losses, training, metrics, final evaluation, Grad-CAM.
- `configs/first_run.json`: first experiment configuration.
- `scripts/`: data verification and sanity tests.
- `data/splits/`: fixed assignments and deterministic class mapping.
- `results/runs/`: run config, history, logs, validation metrics; large/transient run data stays local.
- `checkpoints/`: local latest and best training state.
- `docs/private_reference/`, `notebooks/legacy/`, `archive/`: local historical material excluded from public Git.

The classifier's 512 and 128 hidden dimensions are this project's implementation choice. They are not attributed to the reference paper.

## Phase 3 validation and focal-loss experiment

`python -m scripts.compare_validation_inference` compares normal inference with horizontal-flip probability averaging on the **saved validation partition only** for the two existing best checkpoints. It writes `results/validation_inference_comparison.csv` and JSON files containing per-class metrics and confusion matrices. The experiment rationale and observed error patterns are in `results/validation_error_analysis.md`.

`configs/focal_v1.json` defines one controlled EfficientNet-B0 + CBAM run with the same split, transforms, optimizer, learning rates, and batch size as the weighted baseline. It uses the original 7,009 training references per epoch (no oversampling). For true-class probability `p_y`, its loss is the batch mean of `-alpha_y (1-p_y)^2 log(p_y)`, where `alpha_y` is the Phase 1 square-root inverse-frequency weight recomputed from training counts. This tests whether downweighting easy predictions improves validation macro F1; no improvement is assumed. Launch with `python -m src.train --config configs/focal_v1.json` after `python -m scripts.preflight_focal`.

## Phase 4 frozen-feature validation

`python -m scripts.cache_phase4_features` caches validation probabilities from the three existing best checkpoints and 1,280-dimensional post-CBAM/global-pooling embeddings from the weighted checkpoint for the original train and validation images. The cache is local under `.cache/phase4/`; no test image is loaded. `python -m scripts.evaluate_phase4` then runs a small validation-only probability ensemble grid, class-balanced Logistic Regression and RBF SVM, ExtraTrees, and CNN/ML probability fusion. Candidate tables and confusion analyses are written under `results/phase4/`. These validation comparisons are exploratory model selection; they are not held-out test estimates.

## Phase 5A metadata feasibility probe

`python -m scripts.probe_multimodal` reuses the cached weighted-CNN embeddings and joins only age, sex, and localization by image ID. It fits missing-age imputation, scaling, and categorical encoding on training rows, then evaluates metadata-only Logistic Regression, single-field ablations, full-feature Logistic Regression/SVM/small MLP, and a small validation-only probability-fusion grid. Results and the preprocessing configuration are in `results/phase5_multimodal_probe/`. The probe found no gain of at least 1.5 percentage points over the best image-only validation results, so full multimodal CNN training is not currently justified. The test set remains locked.

## Phase 5B input and MC-dropout audit

`python -m scripts.audit_image_pipeline` inspects original training-image dimensions and a seeded 150-image black-border sample without changing the training pipeline. `python -m scripts.mc_dropout_validation` reuses cached weighted-CNN validation embeddings, enables only classifier Dropout at inference, and compares 10/20/30-pass probability averages with and without horizontal flip. It also performs one aspect-preserving resize plus center-crop diagnostic. Results are in `results/phase5b_inference_audit/`; the held-out test set is not accessed.

## Phase 5C post-hoc correction

`python -m scripts.prior_correction_validation` reuses cached validation probabilities. It computes the natural target prior from original training counts only, then applies a coarse lambda grid for oversampled-model prior correction and a separately labeled exploratory weighted-CE cost correction. Full per-class metrics and confusion matrices are in `results/phase5c_prior_correction/`. The highest validation accuracy rose slightly through more majority-class predictions while macro F1 declined; this is not evidence of balanced improvement. No model is trained and the test set remains locked.

## Accuracy exploration

The [accuracy leaderboard](docs/accuracy_leaderboard.md) keeps the strict lesion-disjoint and exploratory image-level protocols separate. The highest numeric validation accuracy is **90.75%** with a PanDerm/B0 orientation ensemble on the exploratory split, where some validation images share lesions with training. The best strict validation accuracy is **87.69%**. Neither is a final test score. Method details, saved checkpoint locations, and the single-image predictor are in the [PanDerm follow-up](docs/panderm_image_level_followup.md). The [base-paper audit](docs/base_paper_protocol_audit.md) explains why its reported 98.86% is not directly comparable to an untouched original-image test set.

## Reference

A. M. H. Pardede, Solikhun, and Juni Ismail, “Comparative Analysis of EfficientNet-B0 and MobileNetV3-large Architectures for Imbalanced Multiclass Skin Lesion Classification,” *Journal of Image and Graphics*, 2026. DOI: [10.18178/joig.14.4.551-566](https://doi.org/10.18178/joig.14.4.551-566). Its reported performance is not assumed to reproduce on this lesion-grouped split.
