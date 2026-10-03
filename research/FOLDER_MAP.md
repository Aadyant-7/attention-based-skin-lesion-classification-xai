# Project folder map

## Active research

| Path | One-line purpose |
|---|---|
| `README.md`, `NEXT_STEPS.md` | Current entry point and phase plan. |
| `research/README.md`, `research/PHASE1_AUDIT.md` | Infrastructure guide and Phase-1 findings. |
| `research/common.py` | Class/protocol definitions, hashes and atomic writes. |
| `research/models.py` | Six CNN feature-map adapters and optional CBAM. |
| `research/experiment.py` | Run artifact writer; does not train. |
| `research/registry.py` | Historical evidence import and safe run upserts. |
| `research/plots.py`, `research/build.py` | Saved-evidence figures, dataset tables and preservation audit. |
| `research/phase2/` | Protocol, comparison plan, literature generator and handoff. |
| `research/configs/backbone_comparison/` | Five fixed planned configs plus optional B2; unlaunched. |
| `research/literature/` | Verified-field CSV, paper table, citations and evidence notes. |
| `research/tests/` | CPU/saved-evidence infrastructure checks. |
| `results/master_experiment_registry.csv` | Authoritative evidence/run index. |
| `results/datasets/` | Manifest audits and class-count tables/charts. |
| `results/figures/legacy/`, `results/model_comparison/legacy/` | Historical presentation figures, separated by protocol. |
| `results/structured_experiments/<id>/` | Future config/environment/log/history/metrics/predictions/plots. |
| `results/model_comparison/structured/` | Future controlled backbone tables/charts. |
| `results/ablations/`, `results/ensembles/`, `results/final/` | Future attention, fusion and frozen-test packages. |
| `results/audit/`, `results/legacy/` | Verification, environment, immutable inventory and portable evidence. |
| `checkpoints/structured/<id>/` | Future local `best.pt` and `latest.pt`. |
| `research/xai/`, `results/xai/structured/` | Reserved new XAI code and labelled image/heatmap/overlay packages. |

## Historical and local

| Path | One-line purpose |
|---|---|
| `legacy/` | Historical entry point and relocated old navigation. |
| `src/`, `scripts/`, `configs/` | Preserved executable historical code/configs; shared CBAM reused explicitly. |
| `docs/` | Historical audits kept at referenced locations. |
| `legacy/reports/` | Six archived Word/PDF deliverables. |
| Older `results/` and `checkpoints/` subfolders | Historical experiments at original paths; see legacy index. |
| `data/splits/` | Frozen tracked labels/assignments; do not regenerate. |
| `data/raw/HAM10000/` | Ignored original images/metadata; future training needs them, presentation does not. |
| `.cache/`, `.venv/` | Ignored features/weights/source and Python environment. |
| `legacy/archive/`, `legacy/notebooks/`, `legacy/gradcam_outputs/` | Archived local historical materials. |
| `docs/private_reference/` | Local private references kept with historical notes. |
| `.git/` | Git database; active `structured-research`, historical `main`/`accuracy-exploration`. |
| `.gitignore`, `.gitattributes`, `requirements.txt` | Asset exclusions, byte-preservation policy and dependencies. |

No raw-data/checkpoint/cache duplication or directory junctions are needed. Old executable paths remain valid; new run IDs and destinations cannot overwrite history.
