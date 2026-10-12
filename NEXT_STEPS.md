# Current next steps

Updated 11 October 2026. **Training and performance searches are closed.** Historical launch plans are preserved in [the previous next-steps document](legacy/navigation/cleanup_2026-10-11/NEXT_STEPS.md).

## Retained evidence

- **S83 equal-six, FP32 identity:** 94.011976% exploratory validation accuracy / 0.895200837 macro-F1; [frozen methodology](research/FINAL_EXPLORATORY_ENSEMBLE.md).
- **S83 post-development test audit:** 87.558217% / 0.807031200; [result and limitations](research/FINAL_S83_TEST_AUDIT.md). This cohort had already been evaluated.
- **Original S31 strict methodology:** 90.153027% / 0.843486392 validation; 86.759814% / 0.794472645 original test. [Strict validation](research/FINAL_STRICT_VALIDATION_RESULTS.md), [original test](research/FINAL_LOCKED_TEST_RESULTS.md).
- **Latest R203:** 30 epochs completed; 93.213573% accuracy, separate best macro-F1 0.890102. Both fixed fusion comparisons failed to improve retained S83 accuracy. [Closeout](research/convnext_base_original/CLOSEOUT.md).

## Remaining work

12 October: the user requested a pre-test generalization check. See the
[saved-validation review and prepared R202 assessment](research/generalization_review/README.md).
The correction was rejected; no assessment or training has run. The R202 result
would be reported separately from the retained S83 product.

1. Organization cleanup and verification are complete; use [the current evidence index](reporting/evidence/INDEX.md) as the report-facing entry point. Historical artifacts remain at their referenced paths.
2. Reconcile the guide's report format, team details, actual contributions and publication template before drafting. References remain private under `docs/private_reference/`.
3. Arrange saved model comparisons, matched attention results, controlled ensemble results, curves, class scores and confusion matrices by research question; use [the study index](research/STUDY_INDEX.md).
4. Use the completed, numerically/artifact-verified [exact S83 XAI package](results/final_s83_xai/v1/README.md): 12 fixed validation cases, 72 branch maps, HTML gallery and 12-page PDF atlas. Visual layout review is complete for all 12 atlas pages and a representative CBAM PDF. [S80 XAI](results/final_cbam_reporting/v1/xai/) remains historical five-model evidence. No training, performance search or test evaluation occurred.
5. Draft the report and paper after organization is complete; review citations, figure labels, cross-references, layout and plagiarism requirements.

## Reporting boundaries

- Exploratory validation, strict validation and test/audit results are separate evaluations.
- A thematic presentation may summarize unhelpful studies briefly; preserve negative evidence and disclose changed recipes/protocols.
- Separate accuracy-selected and macro-F1-selected checkpoint results. Do not combine maxima from different checkpoints into one result.
- Do not claim universal CBAM improvement, completed ten-fold evaluation, clinical validity or test accuracy equal to the highest validation score.
- No new training, ensemble search or test inference is queued by this document.

[Reporting workspace](reporting/README.md) · [Folder map](research/FOLDER_MAP.md) · [Research navigation](research/README.md)

[Final product handoff](reporting/FINAL_PRODUCT_HANDOFF.md) gives the saved-results/gallery/atlas opening sequence. Prior navigation is archived under `legacy/navigation/s83_xai_2026-10-11/`.
