# Attention-Based Skin Lesion Classification with Explainable AI

B.Tech CSE Minor Project · HAM10000 · seven classes · active branch: **structured-research**.

## Start here

| Open | What it contains |
|---|---|
| [Research workspace](research/README.md) | Active code, result contract and navigation. |
| [Phase 2 handoff](research/phase2/README.md) | Literature, dataset protocol and frozen comparison plan. |
| [Folder map](research/FOLDER_MAP.md) | Active, historical and local folders explained. |
| [Next steps](NEXT_STEPS.md) | Remaining phases and training gates. |
| [Experiment registry](results/master_experiment_registry.csv) | Verified historical evidence and future structured runs. |
| [Dataset table](results/datasets/strict_lesion_disjoint/class_counts.md) | Verified per-class total/train/validation/test counts. |
| [Saved figures](results/figures/figure_index.csv) | Training curves, confusion matrices and class-score charts. |
| [Legacy research](legacy/README.md) | Preserved code/results/reports and previous navigation documents. |

## Current status

Phases 1–2 are complete. No structured model has been trained. Final architecture is open to controlled evidence. Primary validation uses the existing lesion-disjoint split (7,009/1,503/1,503 images); the test stays locked until methodology freeze.

Historical validation leaders: **87.69% strict accuracy**, **0.7837 strict macro-F1** (different combinations), and **90.75% exploratory accuracy / 0.8573 macro-F1**. These are validation-selected observations, not final test scores. Historical comparison figures are not controlled architecture ablations.

New work uses `research/`, `results/structured_experiments/`, `results/model_comparison/structured/`, and local `checkpoints/structured/`. Historical source/results stay at original paths because saved recipes depend on them. Raw images, weights, caches and private references remain Git-ignored.

## Safe commands available now

```powershell
# Verify preservation and refresh saved-evidence tables; no model execution
.\.venv\Scripts\python.exe -m research.build --skip-figures
# Validate the Phase 2 plan and regenerate literature tables (CPU only)
.\.venv\Scripts\python.exe -m research.phase2.prepare
```

To present saved tables/PNG/PDF figures, simply open them; Python, raw images and checkpoints are unnecessary. A clone without ignored historical assets can rebuild evidence with `--portable`. The shared training runner is Phase 3 work, so there is no valid new training command yet. Before any future GPU run we will state its question, reason, exact start/monitor commands and output paths.

The long historical README and two old handoff guides are preserved under [legacy/navigation](legacy/navigation/README.md). The six existing Word/PDF reports remain under `reports/` as historical reports; their earlier fixed-architecture statements are superseded.
