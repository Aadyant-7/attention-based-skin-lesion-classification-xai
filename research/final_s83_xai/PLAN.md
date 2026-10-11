# Frozen S83 explanation plan

Approved purpose: complete the final product's Grad-CAM/XAI package for the exact
S83 equal-six ensemble. No training, checkpoint selection, performance search,
or test inference is part of this task. S80 XAI remains historical evidence.

## Method and cohort

Use `results/final_exploratory_freeze/v1/frozen_method.json` unchanged: S06
ConvNeXt-Tiny, S18 ConvNeXt-Small, S15 DenseNet201, S10 EfficientNetV2-S,
S03 EfficientNet-B0 and S79 ConvNeXt-Tiny + CBAM. Respect each recorded epoch,
selector and SHA-256; FP32, TF32 disabled, identity RGB square224 bilinear
antialias/ImageNet normalization, probability weight 1/6 each.

Explain only 12 images from the original exploratory validation partition.
Before seeing maps, rank image IDs by SHA-256 of
`S83_XAI_v1|seed42|<image_id>`. Take one correct and one incorrect prediction
for each of melanoma and akiec; one image of each other true class regardless
of correctness; then three additional errors, without duplicates. Never replace
a selected case because its heatmap is unattractive or uninformative.

## Grad-CAM and attention

All six branches explain the **saved S83 predicted class**, including errors.
Differentiate `(1/6) * softmax(branch_logits)[target]`: this is the derivative of
the actual ensemble probability with respect to that branch's activations.
Global-average spatial gradients, multiply activations, sum channels, ReLU,
resize bilinearly to224 and normalize per branch for display. Save both raw and
normalized arrays and flag zero/degenerate maps without hiding them.

Use final spatial feature outputs; DenseNet201 uses `features.0.denseblock4`
(before the following BN/ReLU), others use the last `features` module.
Hook the S79 CBAM channel and spatial sigmoid outputs, preserving native spatial
weights, display-resized spatial weights and all channel weights. CBAM weights
are internal gates, distinct from Grad-CAM or lesion segmentation.

Load one model at a time on the GPU. Set eval mode, freeze parameter gradients
and differentiate only through the input/activations. No optimizer exists.
Use run-ID directories so plain and CBAM ConvNeXt-Tiny cannot collide.
The mean of normalized branch maps is labeled a **display composite**, not a
single exact or causal attribution of the ensemble. Maps do not establish
clinical reasoning, localization accuracy or increased classification accuracy.

## Verification and artifacts

Authenticate the five frozen FP32 validation caches and the S79 FP32 prediction
CSV. Their equal mean must reproduce the complete saved S83 probability table.
The older training-time AMP prediction CSVs are not the FP32 reference.
On the 12 selected images, recomputed member/ensemble probabilities must match
the reference within `atol=rtol=1e-5`, and ensemble labels must be identical.
Hash selected inputs; check map dimensions, finiteness, ranges, six unique
branches, exact weights, target classes, PNG/PDF readability and preserved
checkpoint/freeze/test-audit/S80-XAI hashes. No test image is decoded.

Save plan, case index, probability replay CSV, raw maps, overlays, CBAM weights,
12 complete case panels, a 12-page PDF atlas, portable HTML gallery, provenance,
verification and completion receipts under `results/final_s83_xai/v1/`.
Only publish the completion manifest after numerical/artifact checks pass.
Completed artifacts are protected against accidental regeneration.
