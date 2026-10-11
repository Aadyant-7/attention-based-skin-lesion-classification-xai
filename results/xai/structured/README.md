# Structured XAI evidence navigation

Explainability is implemented for an earlier frozen method. This directory remains a navigation entry point; its former empty/future-only description is superseded.

- Completed package: [../../final_cbam_reporting/v1/xai/](../../final_cbam_reporting/v1/xai/).
- Scope: **S80 five-model method**, ten deterministic validation cases, including difficult-class correct/incorrect examples.
- Each case retains per-model Grad-CAM maps/overlays, inputs, confidence/label metadata, panels and the CBAM component's attention weights.
- Method/context: [research/final_cbam_reporting/REPORTING_HANDOFF.md](../../../research/final_cbam_reporting/REPORTING_HANDOFF.md).

These are not exact explanations of the retained **S83 six-model ensemble**. Matching S83 explanations remain future work unless separately generated and verified. No XAI inference is performed during cleanup.

Grad-CAM describes class-related activations; CBAM is learned attention. Neither proves segmentation, clinical reasoning or a causal explanation. An averaged display heatmap is not a jointly trained ensemble explanation, and XAI itself does not raise accuracy.

The [reporting evidence index](../../../reporting/evidence/INDEX.md) identifies explanation figures with their correct method/protocol labels.
