# S89: one new transformer before further ensemble expansion

9 October 2026. **Prepared, awaiting explicit GPU approval; not launched.** Report writing remains paused. S88's conditional joint saved-model fusion was not run because S86/S87 failed.

## Why this experiment

S83's six-member ensemble reached **94.0120% accuracy /0.895201 macro-F1** on the reused exploratory validation cohort. Adding the two unused models with the most complementary hard predictions did not improve its probability fusion: ResNet101 recovered6/lost10; frozen PanDerm Large recovered5/lost9; both93.7458%. Therefore, indiscriminately adding all available predictions is unsupported.

Next candidate: **Swin-T**, fully fine-tuned from the explicit torchvision `Swin_T_Weights.IMAGENET1K_V1` checkpoint. Shifted-window self-attention is a genuinely different feature mechanism from the five existing CNN branches plus CBAM. PanDerm's frozen feature/SVM screens do not constitute a fully fine-tuned transformer experiment.

- [Official architecture and pretrained models](https://github.com/microsoft/Swin-Transformer): hierarchical shifted-window attention, transferable vision representations.
- [Official torchvision Swin-T documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.swin_t.html): exact pretrained enum, 28.3M original ImageNet parameters and approximately4.49GFLOPs. These are ImageNet metadata, not skin-lesion accuracy evidence. Replacing the1000-class head changes the parameter count.
- [Recent HAM10000 transformer ensemble author abstract](https://pubmed.ncbi.nlm.nih.gov/42661729/): uses Swin-Tiny, ViT-Base and DeiT-Small in a heterogeneous ensemble. Abstract-level method relevance only: full protocol could not be retrieved here, and its high reported score is not a prediction for our split. The repository's existing SciSpace review already identifies transformer/fusion work but does not establish directly comparable results.

**Research question:** can a fully fine-tuned shifted-window transformer become a sufficiently strong or complementary new component to improve S83 without sacrificing macro-F1/melanoma recall? This is a hypothesis, not a promised95% result.

## Fixed recipe and bounded screening

- Same7009 train/1503 validation exploratory image-level development manifest and seed42. Test loader absent; no new test evaluation or test-based model selection. This is further post-test exploratory development, not independent confirmation.
- Same224x224 RGB bilinear square resize, ImageNet normalization and training horizontal flips. This deliberately retains the comparison recipe; it does not reproduce torchvision's official resize/crop inference transform.
- Full fine-tuning; seven-class dropout0.2/linear head. Preserve native Swin attention, LayerNorm, patch merging and default stochastic depth0.2. No addedCBAM on this first transformer; existing CBAM branch stays available in S83.
- Training-only normalized sqrt inverse-frequency weighted cross-entropy. AdamW backbone3e-5/head1e-4, weight decay1e-4, microbatch16×accumulation2=effective32. No focal loss, Mixup, oversampling, metadata, TTA or resolution change.
- AMP/GradScaler training with finite-loss checks, skipped nonfinite-gradient updates recorded and clipping norm1; FP32 identity validation.
- ReduceLROnPlateau factor0.5/patience2/absolute accuracy delta0.002; record LR history. Numerical accuracy/F1 checkpoints may update on tiny gains; meaningful plateau counters reset only on cumulative accuracy+0.002 or macro-F1+0.003.
- **Mandatory stop at epoch15 for the fresh launch.** Never automatically extend. Configuration's absolute maximum is20. Resume to20 requires a separate approval and explicit boundary argument. Plateau stopping after min15 additionally requires6 stale epochs and an LR reduction.

After15 compare with S06's actual first15 history: best91.1510% /0.856052 macro-F1. Continuation to20 is worth considering only if the candidate is within0.5percentage points of that accuracy and within0.02 macro-F1, or has a clear late improvement (>=0.5points from its first10-epoch best with >=89.5% accuracy and >=0.82 macro-F1). Also report difficult-class behaviour and complementarity; do not lower these criteria afterward. Borderline/flat cases receive no automatic extension. A frozen-feature or two-sample CPU safety check cannot forecast fine-tuning performance.

Estimated RTX4060 8GB cost: **30–60 minutes for15 epochs**, approximately**3–6GB VRAM** at batch16. These are planning estimates, not GPU measurements. If memory does not fit, stop and document a separately reviewed accumulation amendment; do not silently change the recipe.

## Commands and artifacts

From project root, after approval:

```powershell
.\.venv\Scripts\python.exe -B -u -m research.short_screening.s89_swin_transfer --run --stop-after-epoch 15
```

Manual monitoring from any PowerShell:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\short_screening\swin_transfer_v1\s89_swin_t_none_exploratory_seed42\train.log' -Tail 20 -Wait
```

If interrupted before15, recover the latest committed epoch with the same boundary:

```powershell
.\.venv\Scripts\python.exe -B -u -m research.short_screening.s89_swin_transfer --run --resume --stop-after-epoch 15
```

- Runner/config: `research/short_screening/s89_swin_transfer.py`, `s89_swin_transfer_v1.json`.
- CPU safety preflight: `results/short_screening/swin_transfer_v1/preflight.json`; official downloaded weights remain ignored in `.cache/torch/hub/checkpoints/`.
- Results/logs: `results/short_screening/swin_transfer_v1/s89_swin_t_none_exploratory_seed42/`.
- Checkpoints: `checkpoints/short_screening/swin_transfer_v1/s89_swin_t_none_exploratory_seed42/` — best accuracy, best macro-F1 and atomic latest with optimizer/scheduler/scaler/RNG/meaningful counters.
- Best/F1/final prediction probabilities, metrics, class scores, raw/normalized confusion PNG/PDF, training/LR history and comparison figures saved; registry stage explicitly distinguishes `pilot_complete` from completed20/plateau.

Only if standalone and error analysis support it, test one equal-seven S83+Swin combination. Do not combine all models or sweep weights by default. A second fresh model will be chosen from the Swin evidence, not scheduled now. Once GPU launch/logging/first update/checkpointing are confirmed, stop interacting and give the monitoring command; await the user's return.
