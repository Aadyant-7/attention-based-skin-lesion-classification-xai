# Structured research workspace

## Local location

Use `E:\Minor Project\Project Development`; the former C: location remains a compatibility junction. [Verified migration and artifact paths](migration/LOCAL_DRIVE_MIGRATION.md).

## Current work — 11 October 2026

[R203 completed closeout](convnext_base_original/CLOSEOUT.md): **30 epochs completed**, original exploratory split. Best accuracy **93.213573%** raw epoch20; best macro-F1 **0.890102** EMA epoch27. Two fixed saved-probability CPU fusions did not improve S83 accuracy: addition93.945442%, Small replacement93.812375%. **Retain S83 94.011976% / 0.895201. GPU training closed; drafting next.** No extension or test evaluation queued. Source/config/closeout: `research/convnext_base_original/`; verified artifacts and bounded fusion study: `results/structured_experiments/convnext_base_original_v1/`; local checkpoints: corresponding `checkpoints/structured_experiments/convnext_base_original_v1/`.

R202's completed different-split result is preserved in [its verified closeout](image_level_replication/v2/R202_CLOSEOUT.md). Sections below are chronological historical plans/status; their pending-launch and test-lock statements reflect the dates written and do not supersede the current R203 plan.

## Historical navigation and research contract

**Active priority (3 October2026):** [High-performance heterogeneous ensemble phase](phase3/HIGH_PERFORMANCE_ENSEMBLE_PLAN.md): DenseNet201 next ([S15 proposal](phase3/S15_PROPOSAL.md)), EfficientNet-B3 second subject to evidence/approval, conditional Swin-T third; equal voting then a small bounded global-weight study. **S14 deferred by user; no new experiments launched.** Reference remains S12 92.42%/.8770, FP32 identity92.48%/.8775; S13 TTA rejected. Same exploratory split; strict confirmation postponed; original locked test untouched.

**Local organization cleanup:** archives live under `legacy/`; immutable inventory uses `legacy/path_map.json`. Current verification is `python -m research.verify_layout --phase3-current`, preserving earlier cleanup/preparation audits. See [cleanup report](../legacy/organization/README.md). Historical code, datasets, results and checkpoints remain stable.

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
| `results/structured_experiments/<id>/` | Structured configs, environment, history, metrics, predictions and figures; S01 completed, later runs separately approved. |
| `results/ablations/`, `results/ensembles/`, `results/final/` | Reserved outputs for later paired attention studies, controlled ensembles and frozen-test results. |
| `checkpoints/structured/<id>/` | New local checkpoints, outside all historical checkpoint directories. |
| `research/models.py` | Common seven-class feature-map classifier with optional CBAM for seven torchvision CNNs. |
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

The old config and original S01 launch plan remain historical. Active configs/shared runner follow [the two-stage strategy](phase3/TWO_STAGE_STRATEGY.md) and [next launch proposal](phase3/S15_PROPOSAL.md). Every new GPU run waits for approval. Use the shared runner for comparisons.

Strict v1 selects macro-F1; exploratory screening v1 selects accuracy and saves a separate macro-F1 winner. `Experiment.complete()` verifies the declared criterion, supports, split identity and metrics. Checkpoints remain in `checkpoints/structured/`; explicit epoch resume repairs history/registry and completed IDs exit without training. The final test runner is unimplemented and the test remains locked.

Use `research.plots.metric_figures()`, `training_figures()` and `comparison_figures()` with saved results for raw/normalized confusion matrices, per-class scores/support, available curves and model comparisons. PNG exports are 300 dpi and PDFs are vector figures. A missing historical field/curve is omitted, not synthesized. Every highlighted legacy figure directory includes `provenance.json` and a per-class CSV.

## Research boundaries

- Build scripts read manifest labels for the requested train/validation/test count table, but never open test images or produce test predictions.
- New comparisons use the same split, class order, preprocessing, loss, selection metric and effective batch size. Record intentional exceptions.
- Compare attention against the **same new backbone/head/training recipe** without attention; the old custom B0 head is not a valid isolated CBAM control.
- Final selection uses validation evidence and compute/complexity. Freeze ensemble members/weights and protocol before one locked-test evaluation.
- PanDerm pretraining overlap cannot be independently ruled out; source weights have distribution/licensing limits and remain local.
- Grad-CAM code exists for historical B0 only; adapt/verify the selected model's target layer later. No reviewed XAI examples exist yet.
