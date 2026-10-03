# S03 proposal — matched exploratory EfficientNet-B0 control

**Historical proposal; approved and completed 3 October 2026.** See [S03 closeout](S03_CLOSEOUT.md) and [next CPU proposal](S04_PROPOSAL.md). ID `s03_efficientnet_b0_none_exploratory_seed42`. Original prelaunch text retained below as provenance.

S02 achieved86.03% accuracy/.7861 macro-F1 with a generalization gap. Historical B0+CBAM86.23% changes head,attention,batch and selection. One B0 control under **S02's identical exploratory recipe** supplies a fair architecture comparison, a future CBAM baseline and aligned probabilities for bounded fusion. Improvement is a hypothesis, not a promise of90%.

Question: does B0 outperform MobileNetV3-Large with protocol,head,training and selection held constant? Only backbone/official ImageNet weight specification change. Native SE retained, no added CBAM. Recipe `exploratory_screening_v1`,seed42,224 input,flip,sqrt weights,batch16/accumulation2,AdamW,accuracy primary plus independent macro-F1 winner,20-epoch cap/five stale accuracy epochs. Approximately4.017M parameters.

Existing exploratory split7,009/1,503/1,503 (~70/15/15);563 shared train/val lesions disclosed; locked test excluded. Fresh external-pretrained initialization, not S01 weights. Strict confirmation remains separate later fresh training.

Config `research/configs/phase3/s03_efficientnet_b0_none_exploratory_seed42.json`,SHA `f516d4dd0c95e3f298dc98310a1d819965bacd33b4b1544283706095c9d531c7`. CPU readiness passed; no run/checkpoint/registry row created.

Budget: RTX4060 8GB,approximately15–25 minutes (S01 B0 measured16.77min; S02 MobileNet14.94min). Allow1–2GB checkpoint workspace. Actual time depends on load; cap/early stopping bound cost.

From project root **after approval**:

```powershell
.\.venv\Scripts\python.exe -u -m research.train --config research/configs/phase3/s03_efficientnet_b0_none_exploratory_seed42.json
```

Live monitor once log exists:

```powershell
Get-Content .\results\structured_experiments\s03_efficientnet_b0_none_exploratory_seed42\train.log -Tail 30 -Wait
```

Planned outputs:

```text
checkpoints/structured/s03_efficientnet_b0_none_exploratory_seed42/
  best.pt, best_macro_f1.pt, latest.pt
results/structured_experiments/s03_efficientnet_b0_none_exploratory_seed42/
  train.log, progress.json, history.csv, config/environment/pretraining/record
  primary and macro-F1 validation metrics/predictions, summary, figures/
results/model_comparison/structured/exploratory_screening_v1_seed42/
  comparison.csv and two-model comparison figure after completion
results/master_experiment_registry.csv
```

After completion compare aligned S02/S03 errors/classes. Select the stronger backbone for one matched CBAM or augmentation change; consider fixed equal two-model fusion only if complementary predictions justify it. No automatic follow-up runs. Stop for approval.
