# Attention-Based Skin Lesion Classification with Explainable AI

B.Tech CSE Minor Project · HAM10000 · seven classes · Git branch: **structured-research**.

## Current status

**GPU training and performance searches are closed.** The retained method is **S83**: EfficientNet-B0, EfficientNetV2-S, DenseNet201, ConvNeXt-Tiny, ConvNeXt-Small and ConvNeXt-Tiny + CBAM; equal probability averaging (1/6 each), FP32, identity view, 224 × 224 input.

| Saved evaluation | Accuracy | Macro-F1 | Meaning |
|---|---:|---:|---|
| S83 exploratory validation | **94.011976%** | **0.895200837** | Repeatedly used image-level development cohort; train/validation can share lesions. |
| S83 post-development test audit | **87.558217%** | **0.807031200** | Audit of the previously evaluated test cohort; not a new independent first test. |
| S31 strict validation | 90.153027% | 0.843486392 | Separate lesion-disjoint three-model methodology. |
| S31 original strict test | 86.759814% | 0.794472645 | Original locked-test evaluation of that frozen strict methodology. |

Latest R203 ConvNeXt-V2 Base completed 30 epochs: **93.213573%** exploratory accuracy, with separate best macro-F1 **0.890102**. Two fixed fusions did not improve S83 accuracy, so S83 remains retained. These compare complete training packages and do not isolate an architecture effect.

## Start here

- [Reporting workspace](reporting/README.md) and [current evidence index](reporting/evidence/INDEX.md): what to open for report/paper preparation.
- [Retained S83 method](research/FINAL_EXPLORATORY_ENSEMBLE.md) and [S83 test audit](research/FINAL_S83_TEST_AUDIT.md): exact method and evaluation limits.
- [Research study index](research/STUDY_INDEX.md): work grouped by research question, including negative and superseded studies.
- [Folder map](research/FOLDER_MAP.md), [current next steps](NEXT_STEPS.md), [literature](research/literature/README.md).
- [R203 closeout](research/convnext_base_original/CLOSEOUT.md): latest verified standalone and fusion evidence.

## Main folders

| Folder | Purpose |
|---|---|
| `reporting/` | Current report/paper preparation and curated navigation to original evidence. |
| `research/` | Shared implementation, literature, frozen configurations and completed study records. |
| `results/` | Original metrics, predictions, histories, confusion matrices, figures and master registry. |
| `checkpoints/` | Local saved models and recovery states; excluded from Git. |
| `data/` | Frozen split manifests and local raw datasets. |
| `src/`, `scripts/`, `configs/` | Preserved foundational code/configs; some remain shared dependencies. |
| `legacy/` | Archived notebooks, reports, earlier navigation and organization records. |
| `docs/` | Existing development documentation and private guide/reference material. |
| `.cache/`, `.venv/` | Local derived caches/pretrained assets and Python environment; excluded from Git. |

The master index is [results/master_experiment_registry.csv](results/master_experiment_registry.csv). Its rows include training runs, inference candidates and tabular candidates; row count is not model count. Keep checkpoint selectors and evaluation protocols visible when comparing scores.

## Explainability and reproducibility

[Completed XAI](research/xai/README.md) contains verified Grad-CAM/CBAM evidence for **S80's five-model method**. Exact S83 six-model explanations remain a separate task; S80 maps must not be relabelled as S83. XAI does not raise prediction accuracy.

Code, frozen configurations, results and checkpoint paths remain stable because imports and recorded hashes depend on them. Old dated launch plans are historical records, not instructions to restart training. See [research navigation](research/README.md) and [legacy navigation](legacy/README.md).

The physical project is **`E:\Minor Project\Project Development`**. Its former C: path is a compatibility directory junction; [migration record](research/migration/LOCAL_DRIVE_MIGRATION.md). GitHub contains code and tracked evidence, but not all ignored raw images, checkpoint binaries, caches or private references.

Earlier navigation is preserved unchanged in [the 11 October archive](legacy/navigation/cleanup_2026-10-11/).
