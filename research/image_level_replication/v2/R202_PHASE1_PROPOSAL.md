# R202 — ConvNeXt-V2 Base, V2 fold 00

**Phase 1 prepares and verifies the recipe. GPU training is not authorized by this preparation request.** The planned run is one fresh 50-epoch experiment, with no automatic subsequent model or outer-assessment evaluation. The launch freeze and CPU preflight report record the completed checks and downloaded weight hash.

This proposal supersedes the earlier, unlaunched Tiny+CBAM R202 draft. Its different reserved identifier remains preserved in the historical proposal; no run or checkpoint is reused.

## Choice and research question

Use `convnextv2_base.fcmae_ft_in22k_in1k`: FCMAE pretraining followed by ImageNet-22K and ImageNet-1K fine-tuning. The installed timm definition resolves the author file `convnextv2_base_22k_224_ema.pt` from `https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt`; its published license is CC BY-NC 4.0. The seven-class model has **87,699,975 parameters**. Its native V2 blocks retain Global Response Normalization; this run adds no CBAM.

Question: does a stronger pretrained ConvNeXt representation, careful adaptation and a stable late schedule improve generalization on the unchanged V2 inner-validation cohort? This is a complete training-package comparison against R201, not an isolated attribution of a gain to architecture, resolution, EMA or augmentation.

Repository evidence supports choosing ConvNeXt: historical Tiny reached 91.7498%, Small 92.2156%, and the Tiny+CBAM training package 93.0140%. Those use the older 1,503-image exploratory cohort and cannot establish a directly comparable target on the new 902-image cohort. R201's stopped B3 package reached 85.8093% on the new cohort despite near-perfect training fit. None of these figures establishes a future assessment result.

## Frozen split and interpretation

- `data/splits/image_level_replication/v2/fold_00.csv`: **8,111 training / 902 inner validation / 1,002 outer assessment**; seed 42.
- Manifest SHA256: `c69b7da5441e151a2e2248a987994472091018b8ea9c8069d1177bd50b7ff1aa`.
- Class order: `akiec, bcc, bkl, df, mel, nv, vasc`.
- Natural lesion overlap is already permitted: 353 validation images and 389 outer-assessment images have a training lesion represented. Original image IDs remain distinct; no originals or augmentations are copied between partitions.
- This development run loads only training and inner validation. No outer-assessment images are opened or scored.
- Report as **post-development internal image-level K10-inspired evaluation with lesion overlap**. One development fold is not a completed ten-fold study or lesion-independent evaluation.
- Initialization is external pretrained weights only. Historical HAM10000 checkpoints/predictions cannot join this V2 ensemble because their training includes V2 assessment images. If this experiment earns a later repeat on the original exploratory split, that independent repeat can join the historical same-split ensemble.

## Predeclared 50-epoch recipe

| Component | Setting |
|---|---|
| Head | Native average pool → LayerNorm2d(1024, eps 1e-6) → flatten → dropout 0.2 → linear 7 logits |
| Stochastic depth | Fixed drop-path rate 0.1 in the native ConvNeXt-V2 backbone |
| Input | Full-image RGB resize to 224×224, bicubic; ImageNet mean/std; no validation crop |
| Training augmentation | Independent horizontal/vertical flips 0.5; uniform 0°,90°,180°,270° rotation |
| Sampling | Seeded shuffle, each of 8,111 originals once per epoch; no oversampling |
| Imbalance | Train-only square-root inverse-frequency CE weights, normalized to mean 1 |
| Loss | FP32 weighted CE sum divided by target-weight sum over the complete effective batch, including the final partial batch |
| Optimizer | AdamW, betas 0.9/0.999, epsilon 1e-8, weight decay 1e-4 |
| LR | Head 1e-4, final backbone stage 3e-5; earlier groups ×0.8 per stage; stem ×0.4096 |
| No weight decay | Biases and one-dimensional normalization/GRN scale/offset parameters |
| Warmup | Epochs 1–2: native head only, backbone frozen/eval; epochs 3–5: unfreeze and backbone LR factors 1/3,2/3,1 |
| Late schedule | Epochs 6–50 cosine decay; final-stage LR reaches 1e-7 at epoch 50; group ratios preserved |
| Batch | Microbatch 8 × accumulation 4 = effective 32; two workers; final partial group retained |
| Precision | BF16 forward; FP32 CE, softmax and validation; gradient norm clipped to 1 |
| EMA | Fixed decay 0.999; initialize at the start of epoch 3 after head warmup; update only after accepted full-fine-tuning optimizer updates; separately evaluated |
| Duration | Exactly 50 planned epochs; recorded minimum 50 implements that request; manual stopping overrides it; no automatic early rejection, extension or next run |
| Manual stop | Finish and atomically commit the current epoch, preserve artifacts, record the user stop reason |

Microbatch 8 produces 1,014 microbatches / 254 accumulation groups per epoch, versus R201's 2,376 full optimizer steps. EMA remains on the GPU with FP32 parameters during training and evaluation, adding approximately 0.327 GiB VRAM; only checkpoint snapshots are copied to CPU. CPU preparation does not establish GPU feasibility. If the subsequent approved Phase 2 memory preflight requires microbatch 4 × accumulation 8, record that explicit change in a fresh launch freeze before training while preserving effective batch 32. No silent runtime recipe changes.

Raw and EMA validation are distinct predeclared branches: history columns `val_*` always describe raw weights and `ema_val_*` describe EMA. Save each branch's accuracy and macro-F1 winner. Select `best.pt` from the maximum raw-or-EMA accuracy, earliest epoch; same-epoch ties prefer raw. Apply the same independent rule for `best_macro_f1.pt`. Do not average raw and EMA predictions. Meaningful accuracy/F1 counters (0.002/0.003) independently observe each epoch's maximum raw-or-EMA accuracy and F1, but do not replace raw history or end this fixed-duration run. EMA averages floating parameters and copies all current buffers; this native backbone has no BatchNorm.

No focal loss, Mixup, CutMix, massive balanced replay, heavy color transformations, CBAM, TTA or multi-resolution inference is included. This combines a strong new representation with a restrained recipe rather than the poorly generalizing B3 package.

## Artifacts and commands

Configuration: `research/image_level_replication/v2/convnext_base_fold00.json`.

Results/log: `results/image_level_replication/v2/r202_convnextv2_base_fold00_seed42/`.

Checkpoints: `checkpoints/image_level_replication/v2/r202_convnextv2_base_fold00_seed42/`.

Save committed latest/final raw and EMA states; selected and branch best checkpoints; RNG, optimizer and LR state; complete raw/EMA history; meaningful counters and stop reason; predictions/probabilities; class scores; raw/normalized confusion matrices; training/LR curves and PNG/PDF figures; local and master registry entries. Resume must verify weights, config, source, split and preprocessing signatures.

From PowerShell in the project root, **after explicit GPU approval**, first check memory using disposable synthetic inputs and discarded optimizer state. This tests full-stage BF16 training with resident FP32 EMA and both FP32 validation branches; it reads no assessment images and computes no accuracy:

```powershell
.\.venv\Scripts\python.exe -B -m research.image_level_replication.v2.train_convnext_base --memory-check
```

The training command requires that check to pass for the exact frozen setup:

```powershell
.\.venv\Scripts\python.exe -B -m research.image_level_replication.v2.train_convnext_base --train
```

Manual live monitoring from any PowerShell window:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\image_level_replication\v2\r202_convnextv2_base_fold00_seed42\train.log' -Tail 20 -Wait
```

Recovery from the committed `latest.pt`, retaining the same recipe and epoch budget:

```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
.\.venv\Scripts\python.exe -B -m research.image_level_replication.v2.train_convnext_base --resume
```

Explicit safe stop, if requested later:

```powershell
.\.venv\Scripts\python.exe -B -m research.image_level_replication.v2.train_convnext_base --request-stop --reason 'User requested stopping after the current epoch'
```

The runner completes/commits the current epoch, then closes out. A plain `--resume` preserves an explicit stop. Resuming such a stopped run requires a new instruction and the additional `--allow-user-stop-resume` flag; interruption recovery needs only `--resume`. The committed `latest.pt` is authoritative if a partial epoch or bookkeeping step was interrupted. Completed runs can repair closeout from saved predictions using `--closeout`, without further inference.

Provisional cost: **3–5 hours** for 50 epochs on RTX 4060 8GB; GPU memory and throughput remain unmeasured in CPU-only Phase 1. Reserve at least **8–10 GiB disk**, and use the actual preflight disk estimate before launch. The later GPU launch must confirm logging and a recoverable checkpoint, provide the monitoring command, then stop interacting.

## Phase 1 verification — completed

- Final CPU preflight passed in 12.72 seconds. It loaded the official weights strictly into the 87,699,975-parameter model, checked one head-only and one full-fine-tuning update, EMA, optimizer coverage and seven LR schedule epochs. These temporary CPU states were discarded; no model accuracy was measured.
- Weighted gradient accumulation and the final 15-image group match the complete effective-batch loss. Resume/selection probes include nine rejected corruptions and raw/EMA tie handling; actual raw/EMA model, AdamW, LR and RNG state round-tripped through a temporary 1,403,811,754-byte checkpoint.
- Preservation audit passed for 636 historical files, 588 historical registry rows and 127 checkpoint inventory entries. R201's own frozen config/code/weights signature remains unchanged.
- No CUDA initialization or outer-assessment image decoding occurred. The actual R202 result/checkpoint directories and experiment-registry row remain absent until training launches.
- The exact final signature is recorded in `results/image_level_replication/v2/preparation/r202_convnext_base_preflight.json` and `r202_convnext_base_launch_freeze.json`. An earlier, unlaunched preparation freeze is preserved beside them with amendment metadata.
- Downloaded weights remain local in `.cache/torch/hub/checkpoints/convnextv2_base_22k_224_ema.pt`, SHA256 `dd7a88d111c8af2295cb744f5bba888eca26759984b1cac81804ef876807ce17`; they are excluded from Git. This locally recorded digest verifies subsequent integrity.

**Status:** Phase 1 complete; Phase 2 GPU feasibility/training await approval. GPU memory, throughput and future accuracy remain unmeasured. Primary weight/architecture references: [timm model card](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k), [author implementation](https://github.com/facebookresearch/ConvNeXt-V2).

## Later decisions

Close out this complete package on inner validation first, including raw/EMA comparisons and class-wise behaviour. If it earns another run, prepare an independently initialized original-split replication before using historical ensemble members. Keep both protocols and their registries separate. Do not alter the frozen V2 split using scores or present any development result as final held-out performance.
