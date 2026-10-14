# Clone, presentation and code setup

## 1. Download the consolidated project

Install Git, choose any local drive/folder, then run:

```powershell
git clone https://github.com/Aadyant-7/attention-based-skin-lesion-classification-xai.git
Set-Location attention-based-skin-lesion-classification-xai
git branch --show-current
```

The branch should be `main`. GitHub's **Code > Download ZIP** also supplies the current files, without Git history. Use the default branch rather than an old experiment branch or tag.

## 2. Present saved evidence without installing ML tools

Read the [project README](../README.md) and [final product handoff](../reporting/FINAL_PRODUCT_HANDOFF.md). Open saved CSV/JSON tables, PNG/PDF figures and literature Markdown directly. No GPU, Python, raw images or checkpoints are needed for these files.

```powershell
Start-Process '.\results\final_s83_xai\v1\gallery.html'
Start-Process '.\results\final_s83_xai\v1\S83_GradCAM_Atlas.pdf'
```

Run commands from the clone root; retain the complete gallery folder and case subfolders. GitHub renders Markdown, CSV, PNG and PDF evidence. For the HTML gallery, open the cloned file in a browser; GitHub's source preview does not execute it.

## 3. Check a clone without ML dependencies

With Python installed:

```powershell
python -B -m research.maintenance.verify_repository
```

This standard-library check validates saved files, hashes, registry navigation, Python syntax and gallery resources. It imports no ML framework and does not decode images, load model states, fit a model or evaluate the test. No files are written by default. To save an audit receipt explicitly, use `--output` with a path you choose.

## 4. Optional research-code environment

The original installed environment is **Windows / Python 3.14.4 / PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128 / timm 0.9.16**, with RTX 4060 8 GB. The declared PyTorch/torchvision/scikit-learn/timm versions preserve the research setup; other requirements are not a complete platform lock. Dependency checks passed in the existing environment. This cleanup does not claim a fresh network reinstall or exact numerical equivalence on other hardware.

For code inspection or separately authorized reproduction, create an environment using the recorded Python version:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

This generic install does not guarantee the original CUDA wheel variant. For separately authorized GPU reproduction, install the matching PyTorch/torchvision CUDA build for the recorded environment before installing the remaining requirements. `openpyxl` supports historical Excel literature extraction; it is not needed to view the tracked review or run the clone audit. Installing dependencies alone does not start any model run.

Entry points derive the project root from their source location, so a clone can be on another drive. Some archived helpers contain `.venv/Scripts/python.exe`, old launch commands or Windows font paths; those are historical implementation records, not portable presentation instructions.

## 5. What GitHub includes and excludes

| Included in a normal clone | Kept local / not supplied by GitHub |
|---|---|
| Source code, experiment configs and frozen split manifests | Raw HAM10000/external images and raw metadata under `data/raw/` |
| Registry, portable historical metrics, structured metrics and saved predictions/probabilities | Trained and recovery checkpoint binaries under `checkpoints/` |
| Saved curves, confusion matrices, class scores and comparison PNG/PDFs | Pretrained download/feature/source caches under `.cache/` |
| Exact S83 XAI maps, HTML gallery, atlas and verification receipts | Python environment `.venv/`, live logs and temporary files |
| Literature review, references, SciSpace screening and source caveats | Original SciSpace XLSX and private guide/reference documents |
| Current navigation and historical research evidence | Local report DOCX/PDF drafts and editing ZIP under `reporting/report/draft_v1/` and `draft_v2/` |

The original SciSpace workbook remains local at `research/literature/incoming/SciSpace Literature Review.xlsx`. The [tracked integrated review](../research/literature/scispace_analysis/INTEGRATED_REVIEW.md) supplies the screened conclusions in a clone. Private report drafts can be copied separately for group editing.

## 6. Original checkpoints and data

Use [frozen_method.json](../results/final_exploratory_freeze/v1/frozen_method.json) as the exact six-checkpoint list. Five are under `checkpoints/structured/<run_id>/`; S79 is under `checkpoints/final_cbam_development/v1/`. Copy each selected file to its recorded relative path and verify SHA-256 against the manifest. Copy dataset assets only under their documented local paths, subject to source terms. Do not substitute epochs or class orders.

Original receipts can reference ignored checkpoint/raw/cache paths. The clone audit reports those intentional exclusions, while the full [local preservation gate](../research/maintenance/README.md) expects the assets and original environment. Presenting saved results does not require transferring them. The original workbook's recorded hash is cross-checked against tracked provenance even when its bytes are absent from a clone.

No training, explanation regeneration, ensemble search or held-out evaluation is part of setup. Old runners may write artifacts or consume a test-use guard; do not execute them just to open saved results.

## 7. History and interpretation

The active branch is `main`; archive tags preserve former branch tips. Code/config/results stay at recorded paths. The original machine's C: compatibility junction and E: physical location are local migration details and unnecessary on another laptop.

S83 is 94.011976% on exploratory validation and 87.558217% on its recorded, previously evaluated test cohort. Strict S31 and separate R202/R203 studies retain their own labels. Saved explanations do not establish clinical validity. See [current status](project/STATUS.md), [final audit](project/FINAL_AUDIT.md) and [methodological notes](../research/FINAL_EXPLORATORY_ENSEMBLE.md).
