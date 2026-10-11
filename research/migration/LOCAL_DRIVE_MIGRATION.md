# Local drive migration — 11 October 2026

## Current location

`E:\Minor Project\Project Development`

Open this folder in VS Code / Windows Explorer. The GitHub repository and `structured-research` branch are unchanged. This was a storage migration; no training, inference or experiment was performed.

## Preservation and verification

- Copied all 75,555 files and 6,238 subdirectories, including `.git`, ignored raw data, checkpoints, pretrained weights, `.venv`, caches, reports and results.
- Original file content: **55,386,353,643 bytes (51.58 GiB)**. C: free space increased by **55,538,470,912 bytes (55.54 GB / 51.72 GiB)** after removing the redundant copy. Every copied file passed source/destination SHA-256 comparison before the original physical C: files were removed.
- Git working tree and commit connectivity verified; pre-migration HEAD `77356224dd2aa8f52d85d1501de562b9ae8f6cc4` and existing branches/history preserved.
- Relocated Python imports passed on CPU; active project roots resolve to E:. R202/R203 frozen config, source, split and pretrained signatures remain identical, and their completed checkpoint/result packages are present.
- No historical result, metric, checkpoint, frozen source or experiment registry entry was changed.
- Local receipts and full checksum ledger: `.cache/migration/20261011/` (ignored by Git).

## Compatibility with old paths

The former path, `C:\Users\MARS\Desktop\College\Minor Project\Project Development`, is now a **directory junction to the E: folder**. It contains no second physical project copy. It preserves historical absolute links, existing virtual-environment launchers and this chat's original working-directory reference.

Use E: for new commands. Keep the compatibility junction: do not recursively delete the old path, because it points to the live E: project. The base Python installation stays at `C:\Python314`; moving this project does not require reinstalling it.

The E: folder is already attached to the Codex project. If the app still displays C: as its primary folder, choose E: as primary in the project's folder settings when convenient; the C: alias remains functional.

## Useful commands

```powershell
Set-Location 'E:\Minor Project\Project Development'
.\.venv\Scripts\python.exe -B -c "from research.common import ROOT; print(ROOT)"
git status
Invoke-Item '.\results\master_experiment_registry.csv'
Invoke-Item '.\research\convnext_base_original\CLOSEOUT.md'
Invoke-Item '.\results\short_screening\s83_cbam_f1_addition_cpu'
```

## Where the saved work lives

| Location relative to E: project root | Contents |
|---|---|
| `research/` | Research plans, implemented workflows, literature and closeouts. |
| `results/master_experiment_registry.csv` | Index of experiments and saved metrics. |
| `results/` | Histories, predictions, probabilities, class scores, confusion matrices and figures. |
| `checkpoints/` | All original local model checkpoints, including ignored historical runs. |
| `data/` | Preserved raw data, metadata and split manifests. |
| `legacy/` | Earlier reports, notebooks, Grad-CAM outputs and historical documentation. |
| `.venv/` | Copied Python environment; old embedded launch paths work through the junction. |
| `.git/` | Original repository history, branches and remote configuration. |

Training remains closed. Retained S83 exploratory validation: **94.011976% accuracy / 0.895201 macro-F1**; its historical test audit remains **87.558217%**. Migration changes neither result nor its evaluation protocol.
