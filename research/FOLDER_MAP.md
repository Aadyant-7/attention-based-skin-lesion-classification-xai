# Current project folder map

Physical root: **`E:\Minor Project\Project Development`**. The old C: directory is a compatibility junction. Training is closed; current work is organization and preparation for reporting.

## Root folders and files

| Path | What it contains / why it stays |
|---|---|
| `README.md` | Current method, evaluation summary and starting links. |
| `NEXT_STEPS.md` | Remaining reporting/XAI tasks, with no queued experiments. |
| `requirements.txt` | Recorded Python package requirements. |
| `.gitignore`, `.gitattributes` | Local-asset exclusions and Git file handling. |
| `reporting/` | Current report-facing navigation and curated evidence links. |
| `research/` | Literature, shared implementation, configurations and completed study records; [study index](STUDY_INDEX.md). |
| `results/` | Original result packages and [master registry](../results/master_experiment_registry.csv); historical paths remain stable. |
| `checkpoints/` | Ignored local trained weights, best/F1/latest/final states and recovery evidence. |
| `data/splits/` | Frozen strict, exploratory and separate replication manifests plus class labels. |
| `data/raw/` | Ignored HAM10000 and cleared external-data research assets. |
| `src/` | Foundational model/data/loss/Grad-CAM code; CBAM remains a current dependency. |
| `scripts/` | Preserved early runners/probes/preflight helpers with inter-module dependencies. |
| `configs/` | Three foundational configs still referenced by historical code. |
| `docs/` | Existing development documentation and local private report references. |
| `legacy/` | Archived notebooks, reports, old explanations/navigation and preservation records. |
| `.cache/` | Local derived caches, external pretrained weights/source and verification receipts. |
| `.venv/` | Local Python environment for reopening the project. |
| `.vscode/` | Editor visibility/search preferences; hides generated/local-heavy folders. |
| `.git/` | Complete Git history and branch metadata; do not reorganize internally. |

## Current evidence to open

| Path | Meaning |
|---|---|
| [reporting/evidence/INDEX.md](../reporting/evidence/INDEX.md) | Curated entry point to existing report figures/tables. |
| [research/FINAL_EXPLORATORY_ENSEMBLE.md](FINAL_EXPLORATORY_ENSEMBLE.md) | Exact retained S83 six-model method and validation result. |
| [results/final_exploratory_freeze/v1/](../results/final_exploratory_freeze/v1/) | Frozen checkpoint identities/selectors, class order, preprocessing, weights and audit status. |
| [results/short_screening/s83_cbam_f1_addition_cpu/](../results/short_screening/s83_cbam_f1_addition_cpu/) | S83 validation metrics, predictions, class/confusion/comparison figures. |
| [results/final_exploratory_test_audit/v1/](../results/final_exploratory_test_audit/v1/) | S83 audit of the previously evaluated test cohort. |
| [results/final_strict/v1/](../results/final_strict/v1/) | Original strict-validation methodology and ensemble. |
| [results/final_locked_test/v1/](../results/final_locked_test/v1/) | Original strict first-test evaluation. |
| [results/final_cbam_reporting/v1/xai/](../results/final_cbam_reporting/v1/xai/) | Completed S80 five-model explanations, not S83 explanations. |
| [results/structured_experiments/convnext_base_original_v1/](../results/structured_experiments/convnext_base_original_v1/) | R203 Base run and two rejected fixed fusions. |
| [research/literature/](literature/) | Verified review, references and screened SciSpace input. |

## Historical work and path protection

- `legacy/notebooks/`, `legacy/reports/`, `legacy/gradcam_outputs/`, `legacy/archive/`: physically archived earlier materials.
- `legacy/navigation/`: old handoffs and exact navigation snapshots, including [11 October originals](../legacy/navigation/cleanup_2026-10-11/).
- `legacy/organization/`, `legacy/path_map.json`, `results/legacy/`: historical inventories, approved relocations and integrity evidence.
- Completed research studies remain in existing namespaces because later modules reuse them. [STUDY_INDEX.md](STUDY_INDEX.md) gives their questions/status.
- Original result/checkpoint directories stay in place: registries, frozen methods, configs and signatures contain their paths. The relocation map is not a universal substitute for direct references.

Use indexes to choose report material; preserve negative studies and failed-run evidence. GitHub provides tracked code and portable results; ignored raw data, checkpoints, caches and private references need the local project or a separate transfer. Earlier descriptions remain in [the archived map](../legacy/navigation/cleanup_2026-10-11/research/FOLDER_MAP.md).
