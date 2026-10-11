# Current exact S83 explainability

The authorized current workflow explains the frozen **S83 six-model ensemble**, without changing checkpoints, weights, preprocessing or predictions. [Plan](../final_s83_xai/PLAN.md) · [implementation](../final_s83_xai/runner.py) · [package/status](../../results/final_s83_xai/v1/README.md).

The fixed cohort has **12 original exploratory-validation images**: correct/incorrect melanoma and akiec, the remaining true classes and three additional errors. SHA-256 ranking selects cases before viewing maps; unattractive or degenerate maps are not grounds to replace cases.

Each of the six branches targets the **saved S83 predicted class**, including errors. Grad-CAM differentiates `(1/6) × branch softmax probability[target]`, spatially averages activation gradients, weights channels, sums, applies ReLU and resizes for display. The completed package contains **72 branch CAMs**, raw/normalized arrays, overlays and confidence/target metadata. The S79 **macro-F1-selected epoch-35** checkpoint supplies actual channel/spatial CBAM gates, separate from Grad-CAM.

Inference remains FP32, TF32 disabled, identity RGB 224-square bilinear/ImageNet normalization, equal weights 1/6. One model loads at a time; no optimizer, parameter update, test image or performance search is involved. The package's `verification.json` passed and `manifest.json` records completion: 72 finite nondegenerate CAMs, 12 CBAM cases, 192 readable PNGs, 25 valid PDFs and 12 atlas pages. Maximum branch replay difference is 2.980232e-6; ensemble difference 5.215406e-7; zero changed labels. All 12 atlas pages and a representative CBAM PDF passed visual layout review; see the package's `visual_review.json`.

Open the saved HTML gallery or 12-page PDF atlas after completion. Mean normalized branch maps are labelled **display composites**, not exact causal ensemble attribution. XAI does not improve accuracy, establish clinical reasoning or prove lesion segmentation.

## Historical explanations

[S80 five-model XAI](../../results/final_cbam_reporting/v1/xai/) retains ten validation cases/50 branch maps using weights 0.2 and S79 accuracy epoch 33. It is not the current six-model S83 package. Earlier `src/gradcam.py` and aggressive-workflow code remain historical too.
