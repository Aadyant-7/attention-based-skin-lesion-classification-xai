# S02 — MobileNetV3-Large exploratory screening

**Historical proposal; approved and completed 3 October 2026.** See [S02 closeout](S02_CLOSEOUT.md) and [next proposal](S03_PROPOSAL.md). ID `s02_mobilenet_v3_large_none_exploratory_seed42`. Original prelaunch text below retained as provenance.

Question: does this lightweight ImageNet backbone provide competitive exploratory accuracy and useful minority-class performance within 20 epochs? It adds architecture diversity at **2,978,679 common-head parameters**, connects to the base-paper B0/MobileNet comparison, and may provide complementary errors. S01's generalization gap/melanoma confusion makes extending the same strict run weakly motivated. Improvement is a hypothesis.

Config: `research/configs/phase3/s02_mobilenet_v3_large_none_exploratory_seed42.json`; SHA-256 `8be67ef81576fbc2d0be8a02840e0668fae7d4e88f72789dbd3e9467f2fa2f8b`.

Protocol: existing `exploratory_image_level`, `data/splits/exploratory/image_level_dev_v1.csv`,7,009 train/1,503 val/1,503 locked test, approximately 70/15/15. 563 shared train/val lesions disclosed; test assignments preserved and no test loader. See [strategy](TWO_STAGE_STRATEGY.md).

Exact weights `MobileNet_V3_Large_Weights.IMAGENET1K_V2`; seed 42; 224 RGB/ImageNet normalization, flip.5, sqrt training weights, full fine-tuning, AdamW backbone3e-5/head1e-4, batch 16×accumulation2, AMP/determinism, 20-epoch cap and five stale-accuracy epochs. No added CBAM; native SE retained. Accuracy-selected best plus independent macro-F1-best state/predictions saved. Secondary improvements do not extend the primary accuracy stop rule.

Budget: one RTX 4060 8GB, roughly **10–30 minutes** plus an approximately 22MB pretrained download if uncached; allow 1–2GB for atomic snapshots. S01 measured 16.77 minutes; MobileNet GPU fit/speed remain unmeasured. First epoch logs actual time/allocated VRAM. OOM stops for a documented amendment; no silent fallback.

From project root **after approval**:

```powershell
.\.venv\Scripts\python.exe -u -m research.train --config research/configs/phase3/s02_mobilenet_v3_large_none_exploratory_seed42.json
```

Live monitor in another PowerShell after the log exists:

```powershell
Get-Content .\results\structured_experiments\s02_mobilenet_v3_large_none_exploratory_seed42\train.log -Tail 30 -Wait
```

Read-only readiness:

```powershell
.\.venv\Scripts\python.exe -m research.train --config research/configs/phase3/s02_mobilenet_v3_large_none_exploratory_seed42.json --check
```

```text
checkpoints/structured/s02_mobilenet_v3_large_none_exploratory_seed42/
  best.pt                     accuracy winner
  best_macro_f1.pt             macro-F1 winner
  latest.pt                   full epoch resume incl both winners
results/structured_experiments/s02_mobilenet_v3_large_none_exploratory_seed42/
  config.json, environment.json, pretraining.json, record.json
  train.log, progress.json, history.csv, training_summary.json
  validation_metrics.json, validation_predictions.csv
  validation_metrics_macro_f1.json, validation_predictions_macro_f1.csv
  figures/                    primary CM/class/support/curves PNG+PDF+CSV
results/master_experiment_registry.csv
                              real structured run row on actual launch
results/model_comparison/structured/exploratory_screening_v1_seed42/
                              table after completion; graph after 2 comparable runs
```

Resume explicitly with `--resume`, same code/runtime/config and own trusted latest. Completed IDs exit without GPU. No proposed-run output/checkpoint/registry row exists yet. 24 CPU tests passed, including exploratory test-identity protection, accuracy/F1 selection distinction and ties. Next proposals follow its curves/class errors/cost, not an automatic fixed list. Stop for approval.
