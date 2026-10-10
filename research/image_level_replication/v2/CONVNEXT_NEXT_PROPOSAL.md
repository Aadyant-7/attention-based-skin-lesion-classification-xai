# Next candidate: fresh ConvNeXt-Tiny + CBAM on V2 fold 00

Status: proposed recipe; no new GPU run launched. A separate V2 runner and CPU
launch freeze must be prepared before requesting training approval. Do not point
the historical S79 runner at this manifest or reuse a project-trained checkpoint.

## Why change direction

R201's paper-inspired B3 package has not reproduced the historical ConvNeXt
performance. Its late training accuracy is approximately 99.6%, whereas inner
validation remains approximately 85–86% with validation loss above its early
minimum. This is consistent with poor generalization; it does not establish
which of resolution, dense head, sampling or augmentation caused the gap.
Oversampling gives 38,010 training draws and 2,376 optimizer updates per epoch;
extra exposure and compute have not established extra validation benefit.

Verified historical evidence, on the **older exploratory validation cohort**:

| Run | Accuracy maximum | Macro-F1 maximum | Epochs of maxima |
|---|---:|---:|---|
| S06 ConvNeXt-Tiny | 91.7498% | 0.862831 | 17 / 18 |
| S18 ConvNeXt-Small | 92.2156% | 0.854948 | 17 / 17 |
| S79 ConvNeXt-Tiny + CBAM package | 93.0140% | 0.877708 | 33 / 35 |

Sources: `results/structured_experiments/s06_convnext_tiny_none_exploratory_seed42/validation_metrics.json`,
`results/structured_experiments/s18_convnext_small_none_exploratory_seed42/validation_metrics.json`,
and `results/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/training_summary.json`.
S79 changed augmentation, warmup and duration as well as attention; its gain is
not proof that CBAM alone improved accuracy. These scores are not directly
comparable to R201's new 902-image cohort.

## Research question

Does the strongest demonstrated ConvNeXt training package transfer to the frozen
V2 inner-validation cohort with better generalization and much lower repeated
training exposure than R201? This tests a complete package, not a pure backbone
or CBAM ablation. Do not guarantee 93%, 95% or any outer-assessment score.

## Proposed recipe

- Fresh `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, cached at
  `.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth`; no old HAM10000 weights.
- RGB 224×224 bilinear antialias resize, ImageNet mean/std; identity FP32 validation.
- Native ConvNeXt features → CBAM on final 768-channel spatial map
  (reduction 16, spatial kernel 7) → global pooling → native LayerNorm → flatten
  → dropout 0.2 → linear seven-class logits.
- Two epochs training attention/final linear only with frozen backbone in eval
  mode; then full fine-tuning, retaining the optimizer state.
- AdamW, backbone LR `3e-5`, attention/head LR `1e-4`, decay `1e-4`,
  betas `(0.9, 0.999)`, epsilon `1e-8`; clip gradient norm at 1.0.
- Natural training cohort shuffled once per epoch; training-only square-root
  inverse-frequency class weights, normalized to mean 1; weighted CE numerator
  divided by the target-weight sum over the whole effective batch.
- Horizontal/vertical flips and uniform quarter-turns, training only.
  No 38,010-draw oversampling, focal loss, Mixup or 4,096-unit replacement head.
- Batch 16, accumulation 2, effective batch 32; two workers. With drop-last,
  506 microbatches / 253 optimizer updates / 8,096 images per epoch.
- BF16 training / FP32 loss and validation is proposed to avoid S79's FP16
  overflow issue. Record this numerical amendment explicitly; verify on CPU
  and validate GPU BF16 support before launch.
- Maximum 40 epochs, minimum 20; ReduceLROnPlateau on accuracy after warmup,
  factor 0.5, patience 3, absolute threshold 0.002, minimum LR `1e-7`.
- Meaningful patience 8; accuracy delta 0.002 or macro-F1 delta 0.003.
  Tiny improvements still save raw best checkpoints but do not reset patience.
  After epoch 30, stop a flat/declining eight-epoch plateau after an LR reduction
  has settled for two epochs. No absolute accuracy target as a stopping gate.
- Save raw accuracy winner, raw macro-F1 winner, latest/final checkpoints, RNG,
  optimizer/scheduler state, counters, LR history, all predictions/probabilities,
  per-class metrics, confusion matrices, PNG/PDF curves and both registry rows.
- Implement an in-process stop-request hook for future runs so a user can request
  clean cancellation after a committed epoch without an external process helper.

## Split and evaluation

Keep `data/splits/image_level_replication/v2/fold_00.csv` unchanged:
8,111 train / 902 inner validation / 1,002 outer assessment; seed 42.
Class order: `akiec,bcc,bkl,df,mel,nv,vasc`.
The recorded manifest SHA256 is
`c69b7da5441e151a2e2248a987994472091018b8ea9c8069d1177bd50b7ff1aa`.

Natural lesion overlap already makes 353/902 validation images and 389/1,002
outer-assessment images familiar at lesion level to training. Original image IDs
are disjoint. Retain this split; do not force duplicated originals, move difficult
images, or choose a seed using accuracy. Different lesion photographs can remain
on both sides as the frozen protocol permits. Assess only inner validation during
development; freeze decisions before the outer session. Report this as
**post-development internal image-level evaluation with lesion overlap**, not
strict lesion-independent performance or completed ten-fold cross-validation.

All legacy checkpoints and predictions stay preserved, but cannot be fused into
this study: they have seen images assigned to the new outer-assessment partition.
An eventual V2 ensemble requires independently trained same-fold components.

## Budget and reserved identity

Historical S79 took 3,127.8 seconds for 40 epochs on its smaller training cohort.
At V2 size, a provisional estimate is 60–90 minutes at the 40-epoch cap,
approximately 2–4 GiB total GPU memory on RTX 4060 8GB. Actual throughput must be
measured at launch; no current GPU benchmark was run for this proposal.

Reserved ID: `r202_convnext_tiny_cbam_fold00_seed42`.
Proposed results: `results/image_level_replication/v2/r202_convnext_tiny_cbam_fold00_seed42/`.
Proposed checkpoints: `checkpoints/image_level_replication/v2/r202_convnext_tiny_cbam_fold00_seed42/`.
Those run folders and a launch command are not created yet. Next preparation must
produce a tested isolated runner, explicit config and CPU launch freeze, then
present the exact launch/resume/monitor commands before GPU approval.
