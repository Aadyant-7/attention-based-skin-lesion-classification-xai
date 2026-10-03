# S06 proposal — ConvNeXt-Tiny transfer learning

**Historical proposal; approved, completed and verified.** See [S06 closeout](S06_CLOSEOUT.md), [bounded CPU ensemble results](S07_S09_CLOSEOUT.md) and [next proposal](S10_PROPOSAL.md). Original prelaunch text below preserved. ID `s06_convnext_tiny_none_exploratory_seed42`.

Question: does a larger modern ConvNeXt backbone improve matched exploratory single-model performance and add complementary errors useful for two-/three-model fusion?

S05 found no accuracy gain from added B0 CBAM; S04 demonstrated meaningful fusion improvement. Move to a distinct model package rather than more B0 variants. ConvNeXt modernizes a residual CNN design; our installed implementation uses large-kernel depthwise residual blocks,LayerNorm/GELU and no SE/added CBAM, distinct from the lightweight SE backbones already tested. The [original paper](https://arxiv.org/abs/2201.03545) supports architecture motivation, not a HAM10000 guarantee. [Torchvision0.26](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.convnext_tiny.html) documents the exact `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`:82.52% ImageNet-1K top1,109.1MB download. These are pretraining context, **not our skin-lesion accuracy**.

Why now: one stronger/diverse family tests whether representation capacity and complementary errors offer better returns than attention-only B0 modifications. DenseNet/ResNet remain lower-cost alternatives if evidence/resources justify one later; do not run the old fixed list automatically. More parameters alone do not prove useful diversity or better validation generalization.

Config `research/configs/phase3/s06_convnext_tiny_none_exploratory_seed42.json`,SHA `e268c97e151c2a6f8e33b269e9e22385929ce85218ecbaa89e94292e98d9322f`. Registered **same screening v1** asS02/S03,seed42,input224,flip,sqrt training weights,AdamW3e-5 backbone/1e-4 head,batch16/accumulation2,AMP,full fine-tuning,20 cap/five stale accuracy epochs; save independent F1 winner. Native post-pooling LayerNorm retained and assigned backbone LR. No added CBAM; fresh external pretrained weights,not any prior trained checkpoint.

Common direct224 resize remains unchanged for fairness. Official weights recommend236 resize/224 center crop; we deliberately use our registered preprocessing,not that exact published inference pipeline. This is common-budget transfer screening,not a claim to reproduce ConvNeXt's optimal training/inference recipe. Model defaults/native stochastic depth retained.

Existing exploratory manifest:7,009 train/1,503 val/1,503 locked test (~70/15/15),563 shared train/val lesions disclosed. No test loader. CPU preflight and synthetic224 forward passed; **27,825,511 seven-class parameters** (~6.9× B0). No GPU fit/time or pretrained download performed yet.

Budget: **RTX4060 8GB,roughly30–60 minutes**, plus109MB download if uncached;20 cap/early stop. GPU time/memory are estimates,not measured for this model; allow3GB checkpoint/atomic-save workspace. If fit fails, stop and propose an explicit recipe amendment or lower-cost architecture; no silent batch/precision change.

Start from project root **after approval**:

```powershell
.\.venv\Scripts\python.exe -u -m research.train --config research/configs/phase3/s06_convnext_tiny_none_exploratory_seed42.json
```

Manual monitor once log exists:

```powershell
Get-Content .\results\structured_experiments\s06_convnext_tiny_none_exploratory_seed42\train.log -Tail 30 -Wait
```

Planned locations:

```text
checkpoints/structured/s06_convnext_tiny_none_exploratory_seed42/
  best.pt, best_macro_f1.pt, latest.pt
results/structured_experiments/s06_convnext_tiny_none_exploratory_seed42/
  train.log, progress.json, history.csv, config/environment/pretraining/record
  primary+secondary metrics/predictions, summary, curves/matrices/class figures
results/model_comparison/structured/exploratory_screening_v1_seed42/
  same-policy single-model/attention comparison after completion
results/master_experiment_registry.csv
```

Launch independently after approval; confirm log/first checkpoints, provide monitor/resume instructions,then **STOP**. No epoch polling,background monitoring or completion wait. User returns for closeout. Resume only after original process stops,with the start command plus`--resume` and unchanged launch code/config/runtime.

Completion review examines balanced/class metrics,cost and aligned errors versusS02/S03/S04. Then propose at most two fixed pair ensembles and one fixed triple if complementarity supports them; no weight sweep or automatic additional GPU experiment. See [bounded strategy](DIVERSE_BACKBONE_STRATEGY.md).
