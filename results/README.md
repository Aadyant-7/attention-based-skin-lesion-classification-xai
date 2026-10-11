# Saved research results

Updated 11 October 2026. These are existing outputs; opening CSV, Markdown, PNG or PDF files does not retrain a model. Scientific artifact paths remain unchanged after cleanup.

Start with the [reporting evidence index](../reporting/evidence/INDEX.md), then [master_experiment_registry.csv](master_experiment_registry.csv) for the full history. Failed, screening-only and superseded studies remain preserved; not every registry row is a fair standalone-model comparison.

## Current retained evidence

| Evidence | Saved location | Interpretation |
|---|---|---|
| Retained S83 equal-six ensemble | [short_screening/s83_cbam_f1_addition_cpu/](short_screening/s83_cbam_f1_addition_cpu/) | **94.011976% accuracy / 0.895200837 macro-F1**, original exploratory validation. |
| Exact S83 method/checkpoint freeze | [final_exploratory_freeze/v1/frozen_method.json](final_exploratory_freeze/v1/frozen_method.json) | Six checkpoint paths/hashes, selected epochs, equal weights, preprocessing and class order. |
| S83 recorded test audit | [final_exploratory_test_audit/v1/](final_exploratory_test_audit/v1/) | **87.558217% / 0.807031200**; repeated post-development audit, not a new independent first test. |
| S31 strict validation | [final_strict/v1/](final_strict/v1/) | **90.1530% / 0.843486**; fresh strict B0 + Tiny + V2-S, separate from S83. |
| Original strict held-out evaluation | [final_locked_test/v1/](final_locked_test/v1/) | **86.7598% / 0.794473**; preserved first frozen strict evaluation. |
| Latest R203 standalone and bounded fusions | [structured_experiments/convnext_base_original_v1/](structured_experiments/convnext_base_original_v1/) | Base standalone **93.213573%**; two fixed fusions did not beat S83 accuracy. |
| Separate R202 V2 study | [image_level_replication/v2/](image_level_replication/v2/) | Different 902-image inner-validation cohort; not interchangeable with original-split results. |
| Implemented earlier XAI package | [final_cbam_reporting/v1/xai/](final_cbam_reporting/v1/xai/) | Ten validation cases for the earlier **S80 five-model method**; not exact S83 explanations. |

## Folder map

| Folder/group | Contents |
|---|---|
| [structured_experiments/](structured_experiments/README.md) | Structured training/inference packages, including S01–S30 and the latest original-split R203 family. |
| `model_comparison/structured/` | Matched model, attention and controlled-fusion comparison tables/figures. |
| [ablations/](ablations/README.md), [ensembles/](ensembles/README.md), [final/](final/README.md) | Navigation pointers to actual scientific packages; no duplicated metrics. |
| `short_screening/` | Later bounded screening, fine-tuning, CPU fusion and inference studies, including retained S83 and rejected follow-ups. |
| `architecture_selection/`, `aggressive_enhanced/` | Closed historical architecture/recipe studies, failures, recovery and comparison evidence. |
| `final_cbam_development/`, `final_cbam_reporting/` | S79 attention training, earlier fixed fusion/audit, reporting tables and implemented five-model XAI. |
| `image_level_replication/` | Separate V1/V2 protocol audits and V2 B3/ConvNeXt studies. |
| `datasets/` | Class counts, split/overlap audits and dataset figures. |
| `figures/`, `xai/` | Earlier figure/XAI indexes and structured navigation; newer outputs also live with their experiment packages. |
| `audit/` | Preservation, numerical-recovery and verification receipts. |
| `data_assisted/` | External-data clearance and controlled data-assisted studies. |
| `runs/`, `exploratory/`, `accuracy_exploration/`, `validation_inference/`, `phase4/`, `phase5*/`, `strict_multires_probe/` | Historical local training, metadata/features and inference experiments. |
| [legacy/](legacy/README.md) | Immutable historical inventories, snapshots and small portable evidence copies. |

## What to open

Each serious run normally stores metrics JSON, prediction/probability CSVs, per-class scores, raw/normalized confusion matrices and PNG/PDF figures together. Training runs also retain histories, curves, configuration/environment and logs. Distinguish accuracy-best, macro-F1-best and final checkpoints.

Checkpoints are local under `../checkpoints/` and excluded from Git. Raw images, `.venv`, private references and much historical large-run material are also local-only. Cloning GitHub does not download those assets. Saved tracked figures/tables can be presented without raw data, checkpoints or GPU execution.

Do not move results to improve the visual narrative, relabel validation as test performance, mix protocols, or reinterpret old XAI as exact explanations of a newer ensemble. The reporting layer selects relevant evidence while the full record stays here.
