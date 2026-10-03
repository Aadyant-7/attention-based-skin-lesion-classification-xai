# Presentation handoff guide

> **Historical presentation guide.** The 3-October research direction now permits other architectures and evidence-based final selection. Historical PanDerm scores retain their original protocol labels. Newly generated comparison/CM/training figures are in `results/figures/legacy/` and `results/model_comparison/legacy/`; the old “no saved plots” statement below describes the earlier snapshot. See `research/README.md` and `NEXT_STEPS.md` for current work. The original guide is preserved under `results/legacy/documentation/`.

Use the repository's **`accuracy-exploration` branch**. All scores below are **validation** scores; the locked test set has not been evaluated. The 90.75% result uses an exploratory image-level split with shared lesions and a PanDerm component, so it is not the final EfficientNet + CBAM + Grad-CAM result.

## 1. Where the saved evidence is

| What to show | Exact location | Available after a GitHub clone? |
| --- | --- | --- |
| Trained-run ledger (weighted, oversampled, focal B0 + CBAM) | `results/experiments.csv` | Yes |
| Inference comparison | `results/validation_inference_comparison.csv` | Yes |
| Main accuracy summary and protocol labels | `docs/accuracy_leaderboard.md` | Yes |
| Four-stream B0 + CBAM exploratory table | `results/exploratory/multires_ensemble/summary.csv` | Yes |
| Highest exploratory result, per-class scores, and confusion matrix | `results/exploratory/panderm_base_image_level_tta_v1/summary.csv` and `b0_four_plus_four_svc_views_eight_equal.json` in that folder | Yes |
| Best strict accuracy and its confusion matrix | `results/accuracy_exploration/panderm_base_strict_tta_v1/summary.csv` and `prior_four_with_svc_tta_equal.json` in that folder | Yes |
| Best strict macro-F1 and its confusion matrix | `results/accuracy_exploration/panderm_base_strict_tta_v1/weighted_b0_plus_svc_tta_two_equal.json` | Yes |
| Original run histories and detailed validation metrics | `results/runs/<run_name>/history.csv` and `validation_metrics.json` | **Local only**; `results/runs/` is Git-ignored |
| Best model checkpoints | `checkpoints/<run_name>/best.pt`, including `checkpoints/efficientnet_b0_cbam_weighted_v1/best.pt`; exploratory checkpoints are under `checkpoints/exploratory/` | **Local only**; `checkpoints/` is Git-ignored |
| Research/protocol explanation | `README.md`, `docs/accuracy_exploration_research.md`, `docs/base_paper_protocol_audit.md`, `docs/panderm_image_level_followup.md` | Yes |
| Draft report, layman guide, progress report (Word and PDF) | `reports/` | Yes, on the current `accuracy-exploration` branch |

The result JSON files above contain `confusion_matrix` arrays as well as numeric and per-class metrics. No presentation-ready plot image is saved under `results/`, and `gradcam_outputs/` is currently empty. The Grad-CAM implementation exists at `src/gradcam.py`, but there is no ready overlay to show. Do not claim a saved Grad-CAM figure exists.

## 2. Results worth showing

| Result | Accuracy | Macro-F1 | What to say |
| --- | ---: | ---: | --- |
| Original weighted EfficientNet-B0 + CBAM | 84.56% | 0.7569 | Trained baseline on lesion-disjoint validation; see `results/experiments.csv`. |
| Best strict accuracy: PanDerm/B0 ensemble | **87.69%** | 0.7771 | Lesion-disjoint validation, but PanDerm is outside the approved final architecture. |
| Best strict macro-F1: a different PanDerm/B0 combination | 87.03% | **0.7837** | Same strict protocol; this is *not* the 87.69% configuration. |
| Four B0 + CBAM resolution streams | 88.16% | 0.8280 | Best listed EfficientNet + CBAM exploratory ensemble; image-level split shares lesions. |
| Eight B0/PanDerm streams | **90.75%** | **0.8573** | Highest numeric validation result; exploratory image-level split shares lesions and includes PanDerm. **Not a test result or final model.** |

Show accuracy **and** macro-F1, and keep the split/protocol label beside every score. For a single headline, say: “The highest measured validation accuracy is 90.75% under an exploratory, shared-lesion protocol; the best lesion-disjoint validation accuracy is 87.69%. The untouched test result is pending.”

## 3. Presenting from a friend's laptop (no training)

1. Open [the GitHub repository](https://github.com/Aadyant-7/attention-based-skin-lesion-classification-xai), select the **`accuracy-exploration`** branch, then choose **Code → Download ZIP**. Or run in PowerShell:

   ```powershell
   git clone --branch accuracy-exploration https://github.com/Aadyant-7/attention-based-skin-lesion-classification-xai.git
   cd attention-based-skin-lesion-classification-xai
   ```

2. For a browser-only presentation, open `README.md`, `docs/accuracy_leaderboard.md`, the three `summary.csv` files listed above, and the selected result JSON files. GitHub displays Markdown, CSV, and JSON directly. `src/models.py`, `src/cbam.py`, and `src/gradcam.py` show the method and XAI code.
3. Open this guide and the six current DOCX/PDF files in `reports/` from the clone. Open PDFs in a browser/PDF reader, DOCX files in Word, CSV files in Excel or a text editor, and JSON files in a browser or text editor. No Python setup is needed to read them.
4. The GitHub clone contains saved result tables and result JSON files, **not** the raw HAM10000 images, `results/runs/` histories, local checkpoints, or PanDerm source weights. Bring those local files only if you intend to demonstrate live inference or generate a new Grad-CAM overlay. That would require the correct environment and inputs; for a saved-results presentation, skip it.
5. Raw HAM10000 data is **not needed** to present the saved tables, confusion-matrix numbers, documentation, or source code. Do not run `src.train`, `src.evaluate`, data preparation, or any experiment script just for this presentation.

## 4. Suggested five-minute flow

1. **Repository:** show `README.md` and the branch; explain the seven-class image task.
2. **Methodology:** show the lesion-disjoint split and EfficientNet-B0 → CBAM → classifier; explain that Grad-CAM is a separate explanation step.
3. **Experiment table:** show `results/experiments.csv`, then `docs/accuracy_leaderboard.md` with strict and exploratory rows clearly separated.
4. **Error evidence:** open `confusion_matrix` and `per_class` in the selected JSON result. Say that no prepared heatmap/plot image is currently saved.
5. **Best result:** present 90.75% exploratory and 87.69% strict with their protocol qualifications; mention 88.16% for the EfficientNet + CBAM exploratory ensemble.
6. **XAI and status:** show `src/gradcam.py` as implemented code, without claiming a finished visual review. The final architecture choice, systematic Grad-CAM review, and one locked-test evaluation remain.
