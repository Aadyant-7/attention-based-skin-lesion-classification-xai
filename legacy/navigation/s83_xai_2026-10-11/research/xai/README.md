# Structured explainability implementation

The final S80 five-model workflow now has verified Grad-CAM and actual CBAM attention artifacts for10 deterministic exploratory-validation cases, including correct/incorrect melanoma and akiec. Implementation: `research/final_cbam_reporting/finalize.py`; full method and figure index: `research/final_cbam_reporting/REPORTING_HANDOFF.md`; outputs: `results/final_cbam_reporting/v1/xai/`.

All-five branch maps target the actual0.2-weighted ensemble predicted-class probability. Normalized-map composites are display summaries, not exact causal ensemble attributions. Actual channel/spatial CBAM sigmoid weights are separately saved; XAI does not change predictions or establish clinical validity. Verification confirmed50 finite, nondegenerate CAMs and recomputed prediction agreement; representative panels and attention figures were visually reviewed. No test images were used for these explanations.

Earlier historical `src/gradcam.py` and aggressive-workflow explanation code remain intact; their outputs must not be relabelled as S80 explanations.
