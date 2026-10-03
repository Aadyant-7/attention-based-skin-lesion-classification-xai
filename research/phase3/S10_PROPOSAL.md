# S10 proposal — EfficientNetV2-S transfer learning

**Historical initial proposal; approved and launched. S10 failed during epoch14 validation, with valid epoch13 checkpoints preserved.** See [diagnosis and guarded recovery proposal](S10_FAILURE_RECOVERY.md); original prelaunch text below retained. ID `s10_efficientnet_v2_s_none_exploratory_seed42`.

## Evidence and research question

S06 ConvNeXt-Tiny reached 91.75% accuracy / .8628 macro-F1. Three fixed equal fusions with the weaker S02/S03 models did not increase accuracy; the triple reached .8691 F1 at three-model inference cost. Error diversity exists, but equal averaging cannot reliably recover every complementary correct prediction. Preserve S06 as the accuracy/cost reference and S09 as a balanced alternative.

**Question:** does a stronger Fused-MBConv/SE pretrained package improve common-budget standalone performance and add useful complementary errors to ConvNeXt, enabling a better bounded ensemble?

Recommend **one additional backbone only: EfficientNetV2-S**, rather than a broad list or more B0/CBAM tweaks. The [original EfficientNetV2 paper](https://arxiv.org/abs/2104.00298) introduces training-aware search and Fused-MBConv operations. Its MBConv/SE/BatchNorm design differs from ConvNeXt's depthwise residual LayerNorm blocks, while offering a stronger modern pretrained package than our small B0/MobileNet anchors. Architectural difference is a hypothesis, not proof of useful error diversity. DenseNet/ResNet remain alternatives, not an automatic queue; literature's constrained fusion theme and the observed strength gap make this package a reasonable next use of compute.

[Torchvision 0.26](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.efficientnet_v2_s.html) documents `EfficientNet_V2_S_Weights.IMAGENET1K_V1`: 84.228% ImageNet-1K top1, 82.7MB download and default 384-pixel inference preprocessing. **These are ImageNet context, not HAM10000 results.** We retain registered 224-square preprocessing for common-budget fairness; this does not reproduce the paper's progressive learning or the weights' 384-pixel evaluation. Pretraining recipes and native blocks differ, so this compares model packages, not topology alone.

## Frozen setup and readiness

Config: `research/configs/phase3/s10_efficientnet_v2_s_none_exploratory_seed42.json`; SHA-256 `086a6a5821241fbdb19dec1294e96fbe80562a4d4cb0f94916a69265b0e609be`.

Same `exploratory_screening_v1` policy as S02/S03/S06: seed42, ImageNet normalization, horizontal flip, training-only sqrt class weights, weighted CE, AdamW 3e-5 backbone / 1e-4 head, batch16 with accumulation2, full fine-tuning, AMP, 20-epoch cap/five stale accuracy epochs, independent macro-F1 checkpoint. Common GAP/dropout/linear7 head; the recipe's native ConvNeXt normalization clause applies only to ConvNeXt. V2 native SE and stochastic depth retained; **no added CBAM**.

CPU config/data preflight and synthetic 224-pixel forward passed without CUDA, pretrained download or real-image inference. Seven-class parameters: **20,186,455**. Existing model adapter extended by an explicit channel entry; old launch code remains authenticated through Git.

Unchanged exploratory manifest `data/splits/exploratory/image_level_dev_v1.csv`: 7,009 train / 1,503 validation / 1,503 original locked test (~70/15/15); 563 shared training/validation lesions disclosed. No test loader. Later strict confirmation requires fresh strict training from external weights.

## Budget and commands — after approval

One RTX4060 8GB; **approximately 20–40 minutes** including the 20-epoch cap, subject to early stopping and system load. This is an estimate, not a V2 GPU benchmark. S06 measured 14.94 minutes. Allow roughly 3GB checkpoint/atomic-save workspace and the 82.7MB weight download if uncached. GPU fit has not been tested. If OOM occurs, stop and propose an explicit amendment; no silent batch/input/precision changes.

```powershell
.\.venv\Scripts\python.exe -u -m research.train --config research/configs/phase3/s10_efficientnet_v2_s_none_exploratory_seed42.json
```

Manual monitoring:

```powershell
Get-Content .\results\structured_experiments\s10_efficientnet_v2_s_none_exploratory_seed42\train.log -Tail 30 -Wait
```

Planned locations:

```text
checkpoints/structured/s10_efficientnet_v2_s_none_exploratory_seed42/
  best.pt, best_macro_f1.pt, latest.pt
results/structured_experiments/s10_efficientnet_v2_s_none_exploratory_seed42/
  train.log, progress.json, history.csv, config/environment/pretraining/record
  validation metrics/predictions, class scores, matrices and PNG/PDF curves
results/model_comparison/structured/exploratory_screening_v1_seed42/
results/master_experiment_registry.csv
```

After approval launch independently, confirm the log and first atomic checkpoints, supply manual monitoring/recovery instructions, then **STOP immediately**. No epoch polling, background monitoring or completion wait. Resume only after the original trainer stops, with the same command plus `--resume` and unchanged config/code/environment.

After the user returns: verify both winners, class behaviour, cost and aligned errors. Propose at most one fixed S06/S10 pair first if justified; a third member needs additional evidence. No calibration/weight grid, extra backbone or attention run is automatically authorized. If S10 adds little, move to one justified fixed TTA/resolution intervention rather than continuing a model sweep. CBAM remains available for a useful matched attention question; S05's mixed result does not establish universal benefit or harm.
