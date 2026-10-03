# S05 proposal — matched EfficientNet-B0 + added CBAM

**Historical proposal; approved,completed and verified.** See [S05 closeout](S05_CLOSEOUT.md) and [next ConvNeXt proposal](S06_PROPOSAL.md). ID `s05_efficientnet_b0_cbam_exploratory_seed42`. Original prelaunch plan below retained as provenance.

Question: does adding channel/spatial CBAM before GAP improve B0 accuracy and minority-class performance versus S03 under the matched exploratory recipe? B0 is the structured single-model accuracy leader; S04 confirms fusion helps but minority trade-offs remain. A controlled CBAM pair closes a central methodology/literature gap instead of screening another arbitrary backbone or searching fusion weights. Existing historical CBAM scores changed heads/batches/selection and cannot answer this question. Improvement is a hypothesis, not guaranteed.

Only scientific intervention: **added CBAM**, channel reduction16 and spatial kernel7 on1280×7×7 features before GAP; native EfficientNet SE retained. Same ImageNet V1 initialization (fresh, not S03 trained weights), head/dropout,input224,flip,training-only sqrt weights,AdamW LR policy,batch16/accumulation2,AMP,seed42,accuracy selection/scheduler and20-epoch cap/five stale epochs. Fresh CBAM/head use the existing fresh-parameter LR1e-4; backbone3e-5. Independent macro-F1 winner saved. Common seed/initialization policy does not imply identical random head weights or gradient paths across architectures; one seed limits effect claims.

Registered recipe `exploratory_cbam_v1` is a copy of screening v1 with declared attention/phase/version changes; historical configs remain unchanged. Scientific comparison is S03 versus S05, and their recipe fields must match except those declarations. Future common comparison grouping verifies this equality before including the attention candidate.

Protocol: existing exploratory `data/splits/exploratory/image_level_dev_v1.csv`,7,009 train/1,503 val/1,503 locked test (~70/15/15),563 shared train/val lesions explicitly disclosed. No test loader. Strict confirmation remains later fresh training of selected methods.

Config `research/configs/phase3/s05_efficientnet_b0_cbam_exploratory_seed42.json`, SHA `365769fe768288e78fae623525969f97263df14910a5a5050962f9ce4b8d35e5`. Read-only readiness passed; **29 CPU checks passed**, including CBAM spatial forward/finite gradients/optimizer assignment and recipe-change rejection. Parameter count4,221,413 (~5.1% above S03). No GPU fit/time measured for this exact candidate.

Budget: **RTX4060 8GB, approximately15–30 minutes**,20-epoch maximum/early stopping; S03 B0 measured15.64 minutes. Allow1–2GB checkpoint workspace. Actual speed/memory measured on launch; do not silently change recipe on OOM.

Start from project root **after approval**:

```powershell
.\.venv\Scripts\python.exe -u -m research.train --config research/configs/phase3/s05_efficientnet_b0_cbam_exploratory_seed42.json
```

Live monitor once the log exists:

```powershell
Get-Content .\results\structured_experiments\s05_efficientnet_b0_cbam_exploratory_seed42\train.log -Tail 30 -Wait
```

Planned locations on actual launch:

```text
checkpoints/structured/s05_efficientnet_b0_cbam_exploratory_seed42/
  best.pt, best_macro_f1.pt, latest.pt
results/structured_experiments/s05_efficientnet_b0_cbam_exploratory_seed42/
  train.log, progress.json, history.csv, config/environment/pretraining/record
  primary and macro-F1 validation metrics/predictions, training summary, figures/
results/model_comparison/structured/exploratory_screening_v1_seed42/
  common matched training comparison after completion
results/master_experiment_registry.csv
```

On approval launch independently, confirm log and first checkpoints, provide monitor/recovery commands, then **stop immediately**. No epoch polling, background monitoring or completion wait; user returns for closeout. If interrupted, ensure original process stopped then use the start command plus `--resume` with unchanged launch code/runtime/config.

After completion compare S03/S05 accuracy,F1,class scores/cost and aligned errors before deciding whether a fixed two-model fusion or one targeted augmentation change is worthwhile. Preserve negative results; no automatic follow-up run or weight sweep. Locked test remains untouched.
