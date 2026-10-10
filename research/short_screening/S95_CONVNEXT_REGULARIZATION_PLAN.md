# S95 controlled ConvNeXt-Tiny regularization experiment

10 October 2026. User approved proceeding with the ConvNeXt direction. One20-epoch run; no other candidate, automatic extension or test scoring. CPU preflight passed. This document is a plan, not a result.

## Why this experiment

S06 late training reaches99.7% accuracy versus best91.7498% validation. S79 CBAM also reaches99.9% training while its best validation is93.0140%. This gap suggests a generalization issue but does not establish its cause. S89/S91 transformer additions reduced the actual S83 ensemble performance. Test one change on the proven architecture instead of adding another backbone.

Research question: does stronger AdamW weight decay improve generalization and ConvNeXt-Tiny's contribution to the unchanged S83 ensemble? Official ConvNeXt ImageNet code uses0.05 decay; it is not evidence that this setting is optimal for HAM10000. With our small learning rates, total pure weight shrink is modest; this is a hypothesis, not a promised improvement.

## Fixed setup and comparability

- Fresh cached ImageNet1k `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, hash`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`; full-network fine-tuning, new seven-class head. Original ResearchClassifier, native LayerNorm/stochastic depth, dropout0.2; no added CBAM in this controlled candidate.
- Same exploratory manifest/hash,7009training/1503validation, seed42,224square/ImageNet normalization, horizontal flip only, training-only sqrt inverse-frequency weightedCE, micro16×accumulation2=effective32.
- AdamW backbone3e-5/head1e-4, original parameter-group membership; decay0.0001->0.05 including norm/bias. Same raw-accuracy ReduceLROnPlateau factor0.5/patience2/threshold0 as S06. No focal, oversampling, Mixup or resolution change.
- Exactly20epochs. Meaningful accuracy+.002/F1+.003 counters tracked separately from raw best checkpoints. No patience reset can extend beyond20. Identity FP32 validation and safe AMP-gradient overflow skipping are modern numerical amendments; FP32 can alter checkpoint/scheduler decisions versus historical AMP. Therefore comparison to S06 is informative, not a pure causal weight-decay estimate.
- Existing checkpoints/S83/CBAM/XAI artifacts preserved. Test images/labels excluded from loaders; split integrity checks use IDs/lesion IDs only.

## Evaluation after user returns

Save best accuracy, best macro-F1 and latest full-state checkpoints; history/LR/meaningful counters, optimizer/scaler/RNG recovery, selected/final metrics/probabilities, class scores, confusion matrices, PNG/PDF curves/comparisons and registry entries.

Close out standalone performance and error complementarity first. Exactly one planned equal-six S83 replacement: replace original S06 Tiny with S95's standalone macro-F1 winner, retaining all other members including CBAM and all weights1/6. No weighting/epoch/member sweep. Material gate relative to S83: at least8netcorrect and0.005absolute accuracy gain, macro-F1/melanoma recall nondecreasing. Otherwise retain S83. No final test forecast or scoring.

## Runtime and commands

Estimate15–30minutes and2–3GiB process VRAM on RTX4060 8GB, based on historical S06 timing and1.5GiB peak allocated memory. Actual time may vary. Start from project root:

```powershell
.\.venv\Scripts\python.exe -u -B -m research.short_screening.s95_convnext_regularization --run
```

Monitor from any PowerShell directory:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\short_screening\convnext_regularization_v1\s95_convnext_tiny_wd005_exploratory_seed42\train.log' -Tail 20 -Wait
```

Resume from the last atomically committed epoch, not a partial epoch:

```powershell
.\.venv\Scripts\python.exe -u -B -m research.short_screening.s95_convnext_regularization --run --resume
```

Results/log: `results/short_screening/convnext_regularization_v1/s95_convnext_tiny_wd005_exploratory_seed42/`; checkpoints: `checkpoints/short_screening/convnext_regularization_v1/s95_convnext_tiny_wd005_exploratory_seed42/`. Exact code/config/pretrained hashes and CPU checks: `results/short_screening/convnext_regularization_v1/preflight.json`, `PREDECLARED_PLAN.json`. Once launched and logging/checkpointing confirmed, stop AI interaction until the user returns. Training performs its own standard artifact closeout; no automatic further experiment.
