# Required handoff before every GPU run

No GPU work was started in Phase 2. The shared runner does not exist yet; **do not paste a guessed training command**. Before any preflight, training, inference sweep or XAI GPU job, send a concrete announcement containing:

1. Experiment ID and research question; expected value of the run and why saved evidence is insufficient.
2. Models, recipe version/config hash, protocol, seed, epoch/time/compute limit and stop rule.
3. Exact tested PowerShell start command from the project root; never leave placeholders in the real announcement.
4. Exact monitor command and expected status/epoch fields.
5. Actual checkpoint/result/log paths, resume behavior and what is saved on failure.

## Future output contract

For a real experiment ID, save:

```text
checkpoints/structured/<id>/best.pt           macro-F1-selected state
checkpoints/structured/<id>/latest.pt         safe resumable last state
results/structured_experiments/<id>/
  config.json, environment.json, record.json
  progress.json, train.log, history.csv
  validation_metrics.json, validation_predictions.csv
  figures/                                  raw/normalized matrix CSV+PNG+PDF
                                            class metrics/support CSV+PNG+PDF
                                            training/validation curves PNG+PDF
```

`latest.pt` must include model, optimizer, scheduler, scaler, RNG states, class order, manifest/config hashes and completed-epoch marker. Completed runs must exit without training again. Failure must preserve existing files and log the exception/status. The Phase-1 artifact writer supplies part of this contract; the remaining logging, prediction and resume features must be implemented and checked in Phase 3.

After the announced job has created its files, monitoring is standard PowerShell:

```powershell
# The real announcement substitutes the actual ID into both paths.
Get-Content -LiteralPath '.\results\structured_experiments\<id>\train.log' -Tail 40 -Wait
Get-Content -LiteralPath '.\results\structured_experiments\<id>\progress.json'
```

These are templates, not evidence of a running job. Ctrl+C stops log watching, not the training process. Future preflights must use a separate `preflight_<id>` output folder and must not create a completed model-result row. Git tracks small configs/metrics/tables/figures; checkpoints/raw images/cache remain local. Serious XAI packages go to `results/xai/structured/<id>/` with original image, heatmap, overlay, metadata and provenance; identify any local-only image assets in a public clone.
