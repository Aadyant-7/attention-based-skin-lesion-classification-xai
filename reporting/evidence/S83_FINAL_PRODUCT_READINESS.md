# S83 final product readiness

Completed 11 October 2026. **The retained S83 ensemble now has its own verified
Grad-CAM/CBAM package.** This closes the exact-method XAI gap in the earlier
[dated organization audit](FINAL_READINESS_AUDIT.md); that historical snapshot
remains unchanged.

- Exact six checkpoints, epochs, equal weights, preprocessing and FP32 inference
  preserved. No parameter updates, training, performance search or test inference.
- 12 predeclared original exploratory-validation cases; all seven true classes,
  including correct/incorrect melanoma and akiec and three additional errors.
- 72 finite nondegenerate branch CAMs, 12 actual CBAM gate pairs, 180 NumPy
  arrays, 192 PNGs and 25 PDFs, including a 12-page atlas.
- Complete saved FP32 references reproduce S83's 1503 validation probability
  vectors exactly. Selected replay difference at most 2.98e-6 per branch and
  5.22e-7 after fusion; zero label changes.
- All 12 atlas pages rendered and visually reviewed; a representative CBAM PDF
  rendered and reviewed. These are qualitative examples, not localization scores
  or clinical validation. Mean normalized maps are display composites.
- Checkpoints/freeze and 495 historical S80/strict-test/S83-test artifacts retain
  their hashes. Each of 72 separate branch metadata files matches its case record
  and expected feature tensor shape.
- Master registry now has 594 rows: all 593 previous row values are unchanged;
  one new `explainability_artifact` row has blank accuracy/F1 fields. XAI is not
  a new classification result.

S83 remains **94.011976% / 0.895200837 macro-F1 on exploratory validation** and
**87.558217% / 0.807031200 on the recorded post-development test audit**.
Repeated image-level validation and a previously evaluated audit cohort retain
their stated limitations. The original strict S31 results stay separate.

[Final product handoff](../FINAL_PRODUCT_HANDOFF.md) ·
[XAI closeout](../../research/final_s83_xai/CLOSEOUT.md) ·
[Gallery](../../results/final_s83_xai/v1/gallery.html) ·
[PDF atlas](../../results/final_s83_xai/v1/S83_GradCAM_Atlas.pdf) ·
[Numerical/artifact verification](../../results/final_s83_xai/v1/verification.json) ·
[Visual review](../../results/final_s83_xai/v1/visual_review.json) ·
[Machine-readable readiness](S83_FINAL_PRODUCT_READINESS.json)

Remaining work: final report, paper and presentation writing/formatting. No new
model training or performance experiment is queued.
