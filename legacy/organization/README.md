# Local organization cleanup

## Root after cleanup

```text
Project Development/
  research/                    ACTIVE structured code, configs, literature and phase documents
  results/                     Stable results hub: active outputs separated from legacy studies
  data/                        Original local data and frozen tracked manifests
  checkpoints/                 Stable local weights; NEW runs only under structured/
  legacy/                      Archived reports/notebooks/navigation/material and preservation maps
  src/  scripts/  configs/      Historical executable pipeline retained for compatibility
  docs/                        Historical source audits retained at referenced paths
  .git/  .venv/  .cache/        Git database, environment and existing path-bound caches
  .vscode/                     Project-only clutter/search exclusions
  README.md  NEXT_STEPS.md      Active entry point and existing phase plan
  requirements.txt             Dependencies
  .gitignore  .gitattributes    Local-asset exclusions and byte-preservation rules
```

## Physical moves

| Before | Now | Contents |
|---|---|---|
| `reports/` | `legacy/reports/` | Six existing Word/PDF report files; unchanged. |
| `notebooks/` | `legacy/notebooks/` | Original local notebook tree, including its `legacy/` child; unchanged. |
| `archive/` | `legacy/archive/` | Original empty local directory; preserved. |
| `gradcam_outputs/` | `legacy/gradcam_outputs/` | Original empty old XAI output directory; preserved. |
| `SciSpace Literature Review.xlsx` | `research/literature/incoming/SciSpace Literature Review.xlsx` | Unreviewed workbook; moved intact, not parsed/analyzed. |

Old root directories are removed by moving them, not by deleting their contents. [Per-file relocation ledger](relocations.csv) records original/current paths, byte sizes and hashes. [Before inventory](before_inventory.csv) records 506 non-runtime project files before cleanup; cache/environment/data trees were inventoried by counts/sizes without opening raw images. The workbook and notebook remain local-only, consistent with their unreviewed/historical status.

## Kept in place and why

- `src/`, `scripts/`, `configs/`: historical Python module imports, package-relative root calculations and documented executable commands depend on them; these are small genuine code folders. The active implementation is `research/`, never these old recipes.
- Historical `results/` subfolders and `checkpoints/`: hardcoded producers/predictors, saved configs, experiment registries, checkpoint pointers and figure provenance depend on their exact paths. Moving them would require changing preserved evidence and many historical recipes. They remain clearly classified/indexed as legacy; active results use `structured_experiments/`, comparisons use `model_comparison/structured/`, and new weights use `checkpoints/structured/`.
- `docs/` and private references: historical notes have relative links and are cited from research evidence; retaining a conventional documentation folder keeps those references valid.
- `data/`, `.cache/`, `.venv/`, `.git/`: frozen split paths, feature/weight caches, environment launchers and Git history must remain stable. The editor hides cache/environment/bytecode clutter; Windows Explorer's root no longer includes the four moved archive directories.

No symlinks/junctions or duplicate checkpoints were introduced. Historical document snapshots remain verbatim: old report/notebook paths inside them are translated with [path_map.json](../path_map.json). Current navigation points to the real archive destinations. Original manifest/hash rows, source metrics, registry, checkpoints and saved figures are not rewritten.

## Verification

Run `.\.venv\Scripts\python.exe -m research.verify_layout` from the project root. It checks historical hashes through the relocation map, preservation of the complete pre-cleanup inventory (allowing only documented organization code/navigation edits), registry/figure paths and current Markdown links. It never trains, instantiates a model, loads checkpoint objects, opens test images or parses the SciSpace workbook.

Additional checks use only existing evidence/registry/run-contract unittest classes, explicitly excluding model-interface tests. Results are recorded in `results/audit/organization_verification.json`. Syntax/import checks do not execute historical training/inference entry points. Work stops after organization; no Phase-3 work or literature analysis is included.

**Verified outcome:** 247 historical hashes unchanged; 494 pre-cleanup files byte-identical (including eight moved files); only 12 declared organization/navigation files edited. All 152 unique registry path references, 21 current Markdown files and Python syntax/import checks passed. Nine existing non-model tests passed; model-interface tests were excluded.
