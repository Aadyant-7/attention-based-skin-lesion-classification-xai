# Structured research workspace

**Phase3 prepared; GPU approval pending:** start with [the integrated literature review](literature/scispace_analysis/INTEGRATED_REVIEW.md), [adaptive plan and first start/monitor commands](phase3/PLAN.md), [fixed setup](phase2/EXPERIMENTAL_SETUP.md) and [folder map](FOLDER_MAP.md). The reviewed shared runner is `research/train.py`; no structured experiment has launched. Phase2 documents remain the completed planning snapshot. Old root handoff guides live under `legacy/navigation/`.

**Local organization cleanup:** reports/notebooks/archive/old Grad-CAM directories now live under `legacy/`. The original immutable inventory is resolved through `legacy/path_map.json`; preparation verification is `python -m research.verify_layout --phase3-preparation`, which preserves the separate original cleanup audit. See [cleanup report](../legacy/organization/README.md). Historical executable code, datasets, results and checkpoint locations remain stable.

Current direction (3 October 2026): final architecture is open to evidence. The primary benchmark remains the frozen lesion-disjoint split; the locked test is reserved until Phase 8. Historical image-level scores remain explicitly exploratory. Old reports describe the earlier direction and remain unchanged research history.

## Navigation

| Location | Purpose |
|---|---|
| `results/legacy/artifact_manifest.csv` | Immutable path/size/SHA-256/tracking inventory; all historical experiments preserved in place. |
| `results/legacy/evidence/` | Verified copies of small ignored run configs/histories/metrics for a portable GitHub evidence pack; checkpoints and raw images are not copied. |
| `results/master_experiment_registry.csv` | One machine-readable index of historical evidence and new structured run records. |
| `results/audit/` | Environment, duplicate-content groups, missing references, and build verification. |
| `results/datasets/` | Verified primary/exploratory class tables, manifest overlap audits, stacked count figures and LaTeX table fragments. |
| `results/figures/legacy/` | Automatically generated evidence figures, labelled by protocol and source. |
| `results/model_comparison/legacy/` | Separate strict and exploratory historical comparisons; these are not controlled backbone comparisons. |
| `results/structured_experiments/<id>/` | New config, environment, history, metrics, run record and figures. No new models have been trained yet. |
| `results/ablations/`, `results/ensembles/`, `results/final/` | Reserved outputs for later paired attention studies, controlled ensembles and frozen-test results. |
| `checkpoints/structured/<id>/` | New local checkpoints, outside all historical checkpoint directories. |
| `research/models.py` | Common seven-class feature-map classifier with optional CBAM for six torchvision CNNs. |
| `research/experiment.py` | New run logging/artifact contract; never starts training itself. |
| `research/literature/` | Review schema, historical evidence index, and verification guidance. |
| `NEXT_STEPS.md` | Phases, model shortlist, controls, gates and two-week schedule. |

Rebuild evidence on the original machine (no training, inference, or test-image loading):

```powershell
.\.venv\Scripts\python.exe -m research.build
```

Use `--skip-figures` for a faster registry/audit/table refresh. Figures still include dataset counts in that mode. On a clone without ignored checkpoints/run records, use `--portable`: it retains missing-source registry rows and uses verified small evidence copies for charts. Default mode requires all original historical assets; portable mode reports unavailable ignored assets but still fails on changed files or missing historically tracked files. Never regenerate the immutable manifest to hide a change. Historical paths stay intact because old scripts/checkpoints embed them; no model weights are copied.

## Registry meaning

Rows index **saved evidence**, not independent trained networks. `record_kind` distinguishes training runs, inference candidates and tabular sweep candidates. Some CSV candidates are also summarized in best-result JSONs; compare source selectors and methods rather than counting rows as models. `source_sha256` establishes provenance; `split_sha256` records the manifest identity. Empty fields mean **not recorded**, never zero or a guessed default. Historical training/inference claims follow their audited producers. The missing weighted-run runtime remains blank. Historical decisions describe observed leaders; no entry is selected as the new final model.

`research.registry.upsert()` requires a structured ID, checks unique IDs, uses a writer lock, preserves all other rows and atomically replaces the CSV. `research.build` refreshes legacy rows while preserving structured rows. No old `results/experiments.csv` row is modified.

## New experiment contract

The old config example is a historical proposal. Phase3's first prepared config and executable shared runner are documented in [the launch proposal](phase3/PLAN.md); launch waits for explicit GPU approval. Do not use the legacy B0-specific trainer to compare other backbones.

The shared runner uses `Experiment(config)`, `log_epoch()` and `complete()` to validate/publish the selected validation result. It refuses mismatched support, split changes, inconsistent metrics and accidental overwrites. Checkpoints stay inside `checkpoints/structured/`. Its explicit epoch-boundary resume repairs partially published history/registry from atomic full state; completed IDs exit without training. The final test runner remains unimplemented and the test remains locked.

Use `research.plots.metric_figures()`, `training_figures()` and `comparison_figures()` with saved results for raw/normalized confusion matrices, per-class scores/support, available curves and model comparisons. PNG exports are 300 dpi and PDFs are vector figures. A missing historical field/curve is omitted, not synthesized. Every highlighted legacy figure directory includes `provenance.json` and a per-class CSV.

## Research boundaries

- Build scripts read manifest labels for the requested train/validation/test count table, but never open test images or produce test predictions.
- New comparisons use the same split, class order, preprocessing, loss, selection metric and effective batch size. Record intentional exceptions.
- Compare attention against the **same new backbone/head/training recipe** without attention; the old custom B0 head is not a valid isolated CBAM control.
- Final selection uses validation evidence and compute/complexity. Freeze ensemble members/weights and protocol before one locked-test evaluation.
- PanDerm pretraining overlap cannot be independently ruled out; source weights have distribution/licensing limits and remain local.
- Grad-CAM code exists for historical B0 only; adapt/verify the selected model's target layer later. No reviewed XAI examples exist yet.
