# Final S83 Grad-CAM/XAI closeout

Completed 11 October 2026. The final product is the unchanged **S83 equal-six
ensemble plus post-hoc Grad-CAM and the S79 branch's actual CBAM attention**.
No training, new accuracy experiment or test inference ran.

## Exact model explained

| Member | Selected checkpoint | Epoch | Probability weight |
|---|---|---:|---:|
| S06 ConvNeXt-Tiny | best_macro_f1.pt | 18 | 1/6 |
| S18 ConvNeXt-Small | best_macro_f1.pt | 17 | 1/6 |
| S15 DenseNet201 | best_macro_f1.pt | 14 | 1/6 |
| S10 EfficientNetV2-S | best_macro_f1.pt | 15 | 1/6 |
| S03 EfficientNet-B0 | best.pt (accuracy) | 20 | 1/6 |
| S79 ConvNeXt-Tiny + CBAM | best_macro_f1.pt | 35 | 1/6 |

Exact paths and hashes: [original S83 freeze](../../results/final_exploratory_freeze/v1/frozen_method.json)
and [XAI plan](../../results/final_s83_xai/v1/plan.json). FP32, TF32 disabled,
identity view, RGB square224 bilinear antialias/ImageNet normalization remain
unchanged. The six softmax vectors are averaged before argmax.

## Completed package and checks

- 12 validation cases chosen by the predeclared hash ranking, covering all seven
  classes, correct/incorrect melanoma and akiec, and three further errors.
- 72 branch Grad-CAM maps, raw and normalized; all finite and nondegenerate.
- 12 actual CBAM spatial sigmoid gates (native 7x7 and display 224x224), plus all
  768 channel weights per case. The original broad spatial gates are shown
  without contrast stretching or substituting Grad-CAM for attention.
- 192 PNGs, 25 PDFs, 180 NumPy arrays, branch/case provenance and probability
  replay CSV; a 12-page atlas and an offline HTML gallery.
- All 1503 saved FP32 reference vectors reproduce S83 exactly. Fresh selected
  branch probabilities agree within 2.98e-6; fused probabilities within 5.22e-7,
  with **zero label changes**. These are numerical replay differences, not new
  accuracy measurements.
- Checkpoint/freeze hashes and 495 historical S80/strict-test/S83-test artifact
  hashes remained unchanged. Independent CPU artifact review found no mismatch.
- All 12 atlas pages were rendered and visually reviewed; representative CBAM
  PDF layout was also rendered and checked. Labels, branch identities, targets
  and legends are readable. This is layout QA, not clinical validation.
- Bounded explanation generation took 41.36 seconds, with approximately 420 MiB
  peak allocated GPU memory. Parameters were frozen; no optimizer or updates.

## Open for presentation

1. [Offline gallery](../../results/final_s83_xai/v1/gallery.html): open locally in
   a browser; keep its neighboring case folders when copying it.
2. [Complete PDF atlas](../../results/final_s83_xai/v1/S83_GradCAM_Atlas.pdf):
   portable without checkpoints, raw data or Python.
3. [Case index](../../results/final_s83_xai/v1/case_index.csv),
   [verification](../../results/final_s83_xai/v1/verification.json) and
   [completion manifest](../../results/final_s83_xai/v1/manifest.json).
4. [Final-product handoff](../../reporting/FINAL_PRODUCT_HANDOFF.md): metrics,
   matrices, histories, model identities and reporting boundaries together.

## Interpretation and current scores

All six Grad-CAM maps target the **ensemble's saved predicted class**, including
incorrect cases. The derivative is of each branch's probability contribution
`P_member[target]/6`. The normalized-map mean is a display composite, not an
exact causal ensemble attribution. Branch heatmaps may highlight background as
well as lesion features; no segmentation labels or localization score were
used. CBAM gates are internal attention; these figures do not establish correct
medical reasoning, clinical reliability or universal attention improvement.

XAI leaves S83's saved scores unchanged: **94.011976% accuracy / 0.895200837
macro-F1 on exploratory validation**; **87.558217% / 0.807031200 on its recorded
post-development audit of the previously evaluated test cohort**. The 94.01%
value is not independent test accuracy. Strict S31 remains a separate method.

The earlier [S80 explanation package](../../results/final_cbam_reporting/v1/xai/)
is preserved as historical five-model evidence. This new package closes the
exact-S83 XAI gap noted in the earlier dated readiness audit. Remaining work is
report/paper/presentation preparation, not further model-performance work.

## Commands

From the existing project root, verify saved artifacts and refresh only the new
XAI receipts (no model inference):

```powershell
& '.\.venv\Scripts\python.exe' -B -m research.final_s83_xai.runner --verify
```

The `--prepare` and `--generate` commands are for a fresh reproduction directory;
completed-output guards preserve this finished package. Reproduction requires
the six local checkpoint binaries, original validation images, authenticated
FP32 caches and matching dependencies. Viewing saved PNG/PDF/HTML requires none
of those assets.
