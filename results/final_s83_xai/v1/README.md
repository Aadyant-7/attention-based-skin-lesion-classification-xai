# Exact frozen S83 Grad-CAM and CBAM package

This is the completed current six-model explanation package: **12 original exploratory-validation images** and **72 branch Grad-CAMs** selected by the frozen plan. No training, test image, checkpoint change, weight search or prediction modification is involved.

## Completion and opening

`manifest.json` reports `completed` and `verification.json` reports `passed`: 72 finite nondegenerate CAMs, 12 CBAM cases, 192 PNGs, 25 PDFs and 12 atlas pages. Verification covers source/checkpoint hashes, probability replay, zero changed labels, map finiteness and artifact integrity. Maximum branch/ensemble replay differences are 2.980232e-6/5.215406e-7. All 12 atlas pages and a representative CBAM PDF were rendered and visually reviewed; see `visual_review.json`.

Open:

- `gallery.html`: portable browser gallery; keep it with all case subfolders.
- `S83_GradCAM_Atlas.pdf`: 12-page case atlas.
- `case_index.csv`, `plan.json`: predeclared case selection, source hashes and exact members.
- `selected_probability_replay.csv`: reference/recomputed probabilities.
- `<image_id>/gradcam_panel.{png,pdf}`: six-member panel, input, display composite and method labels.
- `<image_id>/<run_id>/`: raw/normalized CAM arrays, heatmap/overlay, metadata and S79 CBAM gate artifacts.

## Exact method

S06 ConvNeXt-Tiny/F1 epoch 18; S18 ConvNeXt-Small/F1 epoch 17; S15 DenseNet201/F1 epoch 14; S10 EfficientNetV2-S/F1 epoch 15; S03 EfficientNet-B0/accuracy epoch 20; S79 ConvNeXt-Tiny+CBAM/F1 epoch 35. Equal weights 1/6, FP32 / no TF32, identity RGB square224 bilinear/ImageNet normalization. [Original immutable freeze](../../final_exploratory_freeze/v1/frozen_method.json).

Cases are hash-ranked before viewing explanations: correct/incorrect melanoma and akiec, one of each remaining true class and three additional errors. Every branch explains the **saved ensemble predicted class**, even when wrong. Gradients target `(1/6) × branch softmax probability[target]`. Raw and display-normalized maps are preserved; zero maps are flagged, not hidden. Actual S79 channel/spatial sigmoid gates are saved separately from CAMs. Averaging normalized maps produces a **display composite**, not an exact causal explanation.

S83 scores remain **94.011976% / 0.895200837 macro-F1 exploratory validation**, and **87.558217% / 0.807031200** in its previously evaluated test-cohort audit. This package does not generate a new performance score. Maps prove neither lesion segmentation nor clinical validity.

[Implementation/plan](../../../research/final_s83_xai/PLAN.md) · [Final product handoff](../../../reporting/FINAL_PRODUCT_HANDOFF.md). [S80 historical XAI](../../final_cbam_reporting/v1/xai/) uses five members and a different CBAM checkpoint; it remains unchanged.
