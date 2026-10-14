# Attention-Based Skin Lesion Classification with Explainable AI

B.Tech CSE Minor Project | HAM10000 | seven classes | default branch: **main**

## Final retained method and status

**Model training, architecture selection and performance searches are closed.** The retained product is **S83**, a six-model ensemble of independently fine-tuned ImageNet-pretrained networks:

- EfficientNet-B0 (S03, accuracy-selected epoch 20)
- ConvNeXt-Tiny (S06, macro-F1-selected epoch 18)
- ConvNeXt-Small (S18, macro-F1-selected epoch 17)
- DenseNet201 (S15, macro-F1-selected epoch 14)
- EfficientNetV2-S (S10, macro-F1-selected epoch 15)
- ConvNeXt-Tiny + CBAM (S79, macro-F1-selected epoch 35)

Each branch receives RGB 224 x 224 images with ImageNet normalization. Final inference averages the six seven-class **probability vectors equally (1/6 each), using FP32 and the identity view**. There is no TTA, multi-resolution inference, metadata correction or joint training of the ensemble. CBAM is added only to S79. [Frozen checkpoint paths, hashes, preprocessing and class order](results/final_exploratory_freeze/v1/frozen_method.json).

| Saved evaluation | Accuracy | Macro-F1 | Interpretation |
|---|---:|---:|---|
| **S83 exploratory validation** | **94.011976%** | **0.895200837** | 1,413/1,503 correct; repeatedly used image-level development cohort with training/validation lesion overlap. |
| **S83 post-development test audit** | **87.558217%** | **0.807031200** | 1,316/1,503 correct; lesion-disjoint from development, but the cohort had already been evaluated. Not a new independent first test. |
| Separate S31 strict validation | 90.153027% | 0.843486392 | Earlier three-model lesion-disjoint methodology. |
| S31 original strict held-out test | 86.759814% | 0.794472645 | Original test evaluation of that earlier frozen three-model method. |

The 94.01% figure is **exploratory validation accuracy**, not test accuracy. The matched B0 comparison found mixed CBAM effects; the later S79 training package changed attention, augmentation and training duration together. No universal CBAM benefit or clinical validity is claimed.

Separate later research is preserved: R202 achieved 93.413174% / 0.901996079 on its own image-level outer cohort, while R203 reached 93.213573% standalone exploratory accuracy. Neither replaces S83 or constitutes strict confirmation of all six members. See the [study index](research/STUDY_INDEX.md) and [generalization review](research/generalization_review/README.md).

## Clone and present the saved project

```powershell
git clone https://github.com/Aadyant-7/attention-based-skin-lesion-classification-xai.git
Set-Location attention-based-skin-lesion-classification-xai
```

The default `main` branch contains the consolidated research history, source code and tracked evidence. Run commands below from the clone's root; no particular drive or user folder is required.

**Viewing results needs no Python, CUDA, raw HAM10000 data or checkpoints.** Open CSV/JSON files in VS Code/Excel, PNG/PDF figures directly, and the saved HTML gallery in a browser:

```powershell
Start-Process '.\results\final_s83_xai\v1\gallery.html'
Start-Process '.\results\final_s83_xai\v1\S83_GradCAM_Atlas.pdf'
```

Keep the gallery beside its case subfolders. GitHub displays Markdown, CSV, PNG and PDF evidence directly; download/clone the repository to browse the HTML gallery normally. [Detailed setup and local-asset requirements](docs/SETUP.md).

## Where to find the actual results

| What to show or inspect | Repository location |
|---|---|
| Presentation opening sequence and exact final files | [Final product handoff](reporting/FINAL_PRODUCT_HANDOFF.md) |
| Selected tables, figures, comparisons and protocol notes | [Reporting evidence index](reporting/evidence/INDEX.md) |
| Full experiment registry, including negative/partial studies | [results/master_experiment_registry.csv](results/master_experiment_registry.csv) |
| Matched standalone backbone comparisons | [research/MODEL_VS_ACCURACY.md](research/MODEL_VS_ACCURACY.md) and [structured results](results/structured_experiments/) |
| S83 validation metrics, per-class scores and saved probabilities | [S83 validation package](results/short_screening/s83_cbam_f1_addition_cpu/) |
| S83 validation confusion matrix | [Raw counts](results/short_screening/s83_cbam_f1_addition_cpu/figures/confusion_matrix.png) / [row-normalized](results/short_screening/s83_cbam_f1_addition_cpu/figures/confusion_matrix_normalized.png) |
| S83 recorded test-audit metrics, probabilities and class scores | [Audit ensemble package](results/final_exploratory_test_audit/v1/ensemble/) |
| S83 recorded test-audit confusion matrix | [Raw counts](results/final_exploratory_test_audit/v1/ensemble/figures/confusion_matrix.png) / [row-normalized](results/final_exploratory_test_audit/v1/ensemble/figures/confusion_matrix_normalized.png) |
| Training histories and accuracy/loss/macro-F1 curves | [Matched member runs](results/structured_experiments/) and [S79 package](results/final_cbam_development/v1/) |
| Exact final Grad-CAM/CBAM gallery, atlas and receipts | [S83 XAI package](results/final_s83_xai/v1/README.md) |
| Strict validation and original strict test | [results/final_strict/v1/](results/final_strict/v1/) / [results/final_locked_test/v1/](results/final_locked_test/v1/) |
| Literature, verified sources and 50-paper SciSpace screening | [Literature index](research/literature/README.md) |
| Questions, decisions and unsuccessful studies | [Study index](research/STUDY_INDEX.md) |

The registry indexes training, fusion, diagnostic and evaluation records; its row count is not the number of trained neural networks. Accuracy and macro-F1 from different checkpoint selectors must remain separately labelled.

## Project structure

```text
README.md                 Project entry point and current results
requirements.txt          Runtime and research dependency declarations
docs/                     Setup, current status, audit and preserved research notes
reporting/                Report/paper/presentation navigation and source evidence
research/                 Models, runners, configs, literature and study closeouts
results/                  Original metrics, predictions, figures and registries
data/                     Tracked split manifests; raw data remains local
src/ scripts/ configs/    Preserved foundational code and shared dependencies
legacy/                   Archived notebooks/reports/navigation and integrity records
checkpoints/              Local trained/recovery weights; ignored by Git
```

Git configuration and editor preferences also remain at the root. Local `.venv/` and `.cache/` are ignored. Research plans and historical handoffs stay beside their studies because imports, frozen receipts and recorded paths depend on them; [folder map](research/FOLDER_MAP.md). Current human/assistant handoff and remaining work live in [docs/project/STATUS.md](docs/project/STATUS.md).

## Verification and reproducibility

From any clone, a **stdlib-only, read-only audit** checks source syntax, tracked navigation, registry references, final source hashes and saved XAI resources:

```powershell
python -B -m research.maintenance.verify_repository
```

This does not import an ML framework, load images/checkpoints, train, initialize CUDA or rerun test inference. The [full local preservation check](research/maintenance/README.md) additionally requires the original local scientific assets and environment. See the [final repository audit](docs/project/FINAL_AUDIT.md).

The original environment used Windows, Python 3.14.4, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and an RTX 4060 with 8 GB VRAM. Setup details and dependency limitations are documented [here](docs/SETUP.md). A clone contains no raw dataset, checkpoint binaries, pretrained caches, private references or local Word/PDF report drafts. These are not needed to present saved results; actual model execution requires separately transferring the recorded local assets.

Exact S83 XAI is complete: 12 predetermined validation cases, 72 branch Grad-CAM maps and actual S79 CBAM gates. Numerical replay changed zero predictions. Explanation maps do not improve accuracy or establish correct medical reasoning. S80's earlier five-model explanations remain historical.

## Git history and remaining work

`main` is the single active branch. The former `main`, `accuracy-exploration` and `structured-research` tips are preserved by dated `archive/*-2026-10-14` tags, alongside `strict-baseline-snapshot`. Consolidation preserves the original commit graph and scientific artifacts.

The editable full report and two-column paper draft have been produced locally and remain private/ignored. Remaining work is group/guide review, verified contributions/signatures, final report/paper formatting and presentation/submission. No new experiment or test evaluation is queued. [Current status](docs/project/STATUS.md).
