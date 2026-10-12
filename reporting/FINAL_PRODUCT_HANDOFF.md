# Final product handoff

Physical project: `E:\Minor Project\Project Development`; branch `structured-research`. Training and performance searches are closed. This guide opens saved evidence; it does not retrain, regenerate explanations or evaluate the test.

## Retained framework

**S83 equal-six probability ensemble:** ConvNeXt-Tiny (S06/F1 epoch 18), ConvNeXt-Small (S18/F1 epoch 17), DenseNet201 (S15/F1 epoch 14), EfficientNetV2-S (S10/F1 epoch 15), EfficientNet-B0 (S03/accuracy epoch 20), ConvNeXt-Tiny+CBAM (S79/F1 epoch 35). Separately fine-tuned ImageNet-pretrained models; FP32, TF32 disabled, identity RGB 224/ImageNet preprocessing; weights 1/6. [Exact checkpoint paths/hashes/preprocessing](../results/final_exploratory_freeze/v1/frozen_method.json).

| Saved evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| S83 original exploratory validation | **94.011976%** | **0.895200837** |
| S83 repeated post-development test audit | **87.558217%** | **0.807031200** |
| Separate R202 EMA20 image-level outer fold00 | 93.413174% | 0.901996079 |
| Separate S31 strict validation | 90.153027% | 0.843486392 |
| Original S31 strict held-out test | 86.759814% | 0.794472645 |

Exploratory development can share lesions and reused validation extensively. The S83 audit cohort was previously evaluated. Do not call 94.01% test accuracy or merge these protocols. R203 Base did not improve S83 through its two fixed fusions.

The later [R202 outer assessment](../research/generalization_review/R202_OUTER_CLOSEOUT.md)
uses a separate model/cohort: 1,002 images with no original image overlap with
R202 training/selection; natural lesion overlap is permitted. Its 93.41% is not
S83's ensemble accuracy, strict performance or a completed ten-fold result.

## What to open

| Purpose | Actual repository path |
|---|---|
| Overall evidence and tables/figures | [reporting/evidence/INDEX.md](evidence/INDEX.md) |
| Complete method history | [results/master_experiment_registry.csv](../results/master_experiment_registry.csv) |
| Retained validation metrics, class scores, probabilities, confusion figures | [results/short_screening/s83_cbam_f1_addition_cpu/](../results/short_screening/s83_cbam_f1_addition_cpu/) |
| Recorded S83 test audit and comparison | [results/final_exploratory_test_audit/v1/](../results/final_exploratory_test_audit/v1/) |
| Separate strict evidence | [results/final_strict/v1/](../results/final_strict/v1/) and [results/final_locked_test/v1/](../results/final_locked_test/v1/) |
| Current exact six-model XAI | [results/final_s83_xai/v1/README.md](../results/final_s83_xai/v1/README.md) |
| Literature and source caveats | [research/literature/README.md](../research/literature/README.md) |
| Research questions/conclusions, including negative studies | [research/STUDY_INDEX.md](../research/STUDY_INDEX.md) |

Exact S83 XAI has completed with passed numerical/artifact verification in `results/final_s83_xai/v1/manifest.json` and `verification.json`: 12 predeclared validation cases, 72 branch CAMs, actual epoch 35 CBAM gates and zero changed labels. Open `gallery.html` for browser browsing or `S83_GradCAM_Atlas.pdf` for the 12-page atlas. All 12 atlas pages and a representative CBAM PDF passed visual layout review; see `results/final_s83_xai/v1/visual_review.json`. Older ten-case S80 five-model maps remain historical evidence. XAI changes no accuracy result; composites are display summaries.

## Practical opening sequence

1. Open this repository in VS Code/Explorer; read the evidence index.
2. Show the model/ensemble comparison, retaining protocol and checkpoint-selection labels.
3. Show S83's saved confusion matrix, class scores and validation metrics.
4. Open the verified exact S83 gallery/atlas; discuss both correct and incorrect cases.
5. Show the separately labelled strict/test-audit results and limitations.

```powershell
Set-Location 'E:\Minor Project\Project Development'
Get-Content '.\results\final_s83_xai\v1\verification.json'
Start-Process '.\results\final_s83_xai\v1\gallery.html'
Start-Process '.\results\final_s83_xai\v1\S83_GradCAM_Atlas.pdf'
```

These commands open saved artifacts only. On another laptop, clone/download the `structured-research` branch and open the same paths relative to its project folder. Keep the gallery and its case folders together. Saved tracked tables, PNG/PDFs and HTML need **no raw HAM10000 data, checkpoints, CUDA or Python** to present.

## Checkpoints and remaining work

Checkpoint binaries are local, Git-ignored: five S83 members live under `checkpoints/structured/<run_id>/`; S79 lives under `checkpoints/final_cbam_development/v1/s79_convnext_tiny_cbam_exploratory_seed42/best_macro_f1.pt`. Use the freeze for exact filenames, selectors and hashes. GitHub alone does not supply checkpoint binaries, raw data, the virtual environment or private guide references. Copy those only if actual inference is required; viewing reports does not require them.

Next work is final report/paper/presentation composition, citation/figure review and guide formatting. Preserve source metrics and meaningful unsuccessful comparisons; a thematic research narrative need not reproduce daily chronology.

[Current final-product readiness](evidence/S83_FINAL_PRODUCT_READINESS.md) · [XAI closeout](../research/final_s83_xai/CLOSEOUT.md).
