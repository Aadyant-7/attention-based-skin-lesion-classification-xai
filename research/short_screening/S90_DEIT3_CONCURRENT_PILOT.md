# S90: authorized concurrent DeiT III Small pilot

9 October2026. The user explicitly requested one new useful model to train alongside the existing S89 Swin run. This authorizes this bounded launch; it does not authorize another queue, test evaluation or automatic continuation.

## Choice and research question

**Exact model:** `deit3_small_patch16_224.fb_in22k_ft_in1k`, author ImageNet22k pretraining followed by ImageNet1k fine-tuning. New seven-class head; full fine-tuning. It is a globally attending patch/CLS-token transformer rather than Swin's shifted-window hierarchy or another CNN variant. Existing PanDerm results used frozen features and SVM heads, so this is a new training architecture/method.

[Model-maintainer card](https://huggingface.co/timm/deit3_small_patch16_224.fb_in22k_ft_in1k) identifies22.1M original parameters,224 input and4.6GMACs; the seven-class head changes the count. [Author paper](https://arxiv.org/abs/2204.07118) supports strong supervised ViT pretraining and transfer. These sources justify a bounded experiment; they do not establish HAM10000 accuracy or guarantee an ensemble improvement. S59 already demonstrated that larger pretraining alone is insufficient.

**Question:** does fully fine-tuned global patch attention produce strong standalone predictions and recover useful errors of S83/Swin, improving a later controlled ensemble without losing more correct predictions?

## Recipe and resource safeguards

- Same exploratory manifest hash,7009train/1503validation, seven-class order and seed42 as S89/S83. No test loader, test images, labels or new test scoring. Reused validation/post-test development remains a limitation.
- Same224 RGB square bilinear/ImageNet normalization, horizontal training flip, training-only normalized sqrt inverse-frequency weightedCE, AdamW backbone3e-5/head1e-4, weight decay1e-4 and dropout0.2 head.
- Preserve pretrained CLS attention, LayerNorm and LayerScale. Explicit stochastic depth0.1; no additionalCBAM, focal loss, oversampling, Mixup, TTA or inference augmentation.
- **Microbatch8, accumulation4, effective32**. Weighted CE uses the combined target-weight denominator across all four microbatches. This changes the microbatch/dropout realization relative to S89; it is recorded, not described as a pure paired architecture ablation. No BatchNorm-dependent microbatch statistics in this transformer.
- AMP/GradScaler training, finite-loss/gradient handling, norm1 clipping, FP32 validation. Skipped scaler updates are recorded. Raw accuracy and macro-F1 winners remain separate, earliest ties.
- ReduceLROnPlateau factor0.5/patience2/absolute delta0.002; meaningful counters require cumulative accuracy+0.002 or macro-F1+0.003. Plateau patience6 after min15 and an LR reduction.
- **Fresh run stops at15 regardless of score.** Hard maximum20 only for a later explicitly approved resume. No automatic ensemble or next model.
- Initial observed RTX4060 memory:8188MiB total,3015MiB used,4942MiB free,54% GPU utilization. This is a snapshot, not a guaranteed peak.
- Before model allocation require at least3072MiB free. Limit this process's PyTorch allocator to40% device memory (~3275MiB). S89 configuration/model/optimizer/precision/checkpoints remain unchanged. If the second process cannot fit, it fails safely with its own evidence; never terminate or shrink S89.
- Expected additional allocation roughly1.5–3GiB; estimated15-epoch wall time40–90minutes while sharing the GPU. These are unmeasured estimates. Concurrency can slow both jobs and does not guarantee a faster combined finish.

## Launch, monitor, recovery

From project root, the authorized command is:

```powershell
.\.venv\Scripts\python.exe -B -u -m research.short_screening.s90_deit3_transfer --run --stop-after-epoch 15
```

Manual monitor from any PowerShell:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\short_screening\deit3_transfer_v1\s90_deit3_small_none_exploratory_seed42\train.log' -Tail 20 -Wait
```

If interrupted before15:

```powershell
.\.venv\Scripts\python.exe -B -u -m research.short_screening.s90_deit3_transfer --run --resume --stop-after-epoch 15
```

- Runner/config: `research/short_screening/s90_deit3_transfer.py`, `s90_deit3_transfer_v1.json`.
- CPU preflight and launcher logs: `results/short_screening/deit3_transfer_v1/`.
- Run metrics/logs/probabilities/history/class scores/raw-normalized confusion PNG/PDF/curves/comparison figures: `results/short_screening/deit3_transfer_v1/s90_deit3_small_none_exploratory_seed42/`.
- Best accuracy, best macro-F1 and atomic latest model/optimizer/scheduler/scaler/RNG/counters: `checkpoints/short_screening/deit3_transfer_v1/s90_deit3_small_none_exploratory_seed42/`.
- Master registry records running/pilot_complete stages through its existing locked atomic upsert.

After first optimizer update, log and checkpoint confirmation, stop interaction and let the processes run independently. On return, close out both pilots separately, compare their same-stage curves and actual class/complementarity results, and only then propose any continuation or fusion. Use S89's documented screening criteria as a reference, not an absolute accuracy promise; no retrospective gate lowering and no indiscriminate all-model average.
