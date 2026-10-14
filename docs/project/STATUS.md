# Current project status and handoff

Updated 14 October 2026. The single active/default branch is `main`. Training, architecture selection and performance searches are closed; no experiment or test rerun is queued.

## Final product

- **S83:** six independently fine-tuned models with equal FP32 identity-view probability averaging at RGB 224 x 224. Only the S79 ConvNeXt-Tiny branch adds CBAM. [Exact method](../../research/FINAL_EXPLORATORY_ENSEMBLE.md) and [frozen identities](../../results/final_exploratory_freeze/v1/frozen_method.json).
- **Exploratory validation:** 94.011976% accuracy / 0.895200837 macro-F1; 1,413/1,503 correct. Reused image-level development cohort with train/validation lesion overlap.
- **Recorded S83 test audit:** 87.558217% / 0.807031200; 1,316/1,503 correct. Lesion-disjoint from development, but previously evaluated. [Audit limits](../../research/FINAL_S83_TEST_AUDIT.md).
- **Exact S83 XAI is completed:** 12 predefined validation cases, 72 branch maps and actual CBAM gates, with zero changed predictions. [Saved gallery/atlas and receipts](../../results/final_s83_xai/v1/README.md).
- **Separate evidence:** [S31 strict validation](../../research/FINAL_STRICT_VALIDATION_RESULTS.md), [its original test](../../research/FINAL_LOCKED_TEST_RESULTS.md), [R203](../../research/convnext_base_original/CLOSEOUT.md), and [R202 outer assessment](../../research/generalization_review/R202_OUTER_CLOSEOUT.md) retain their own protocols and are not S83's test score.

## Work completed

1. HAM10000 loading, class mapping, frozen strict/exploratory split manifests and train-only imbalance controls.
2. Literature integration, matched pretrained backbone screening, attention comparisons and controlled heterogeneous ensembles.
3. Preserved useful, neutral, unsuccessful, partial and failed studies with selectors and protocol labels.
4. Frozen S83 validation, recorded cohort audit, class-wise scores, probabilities, raw/normalized confusion matrices and publication figures.
5. Exact-method Grad-CAM and actual CBAM outputs, including incorrect predictions.
6. Local organization and drive migration with historical results/checkpoints preserved.
7. Full editable report and compact two-column paper drafts, with 30 references, saved privately under `reporting/report/draft_v2/` (local-only, not included in a clone). Word: 81 reviewed pages, 27 embedded figures, native tables/equations; paper: eight reviewed pages. These are review drafts, not an approved submission.
8. Git consolidation and clone documentation; [final audit](FINAL_AUDIT.md).

## Remaining work

1. Review the drafts with the group and guide; enter actual contributions and approved certificate/signature details.
2. Apply final university and publication-venue requirements, confirm citations, refresh page references and export the edited report.
3. Prepare the plagiarism-check copy under the guide's required exclusions and obtain the external similarity report.
4. Complete the presentation and university submission. The communicated upload deadline is 20 October 2026; guide review/plagiarism checking was requested by 14 October.

## How to resume work here

Start with the [main README](../../README.md), [setup guide](../SETUP.md), [final product handoff](../../reporting/FINAL_PRODUCT_HANDOFF.md), [evidence index](../../reporting/evidence/INDEX.md), [study index](../../research/STUDY_INDEX.md) and [literature](../../research/literature/README.md). These are also the current context for future assistants. Archived launch plans describe earlier authorization and do not schedule new work.

Historical reports, plans, navigation and code remain in `legacy/`, `docs/` and the original research namespaces. No generic agent-instruction files needed relocation; the root now contains the project README, requirements and standard configuration only. The previous root status is preserved through Git rename history; older snapshots remain in [the navigation archive](../../legacy/navigation/cleanup_2026-10-11/).

## Reporting boundaries

- Keep exploratory validation, strict validation, original strict test and later audits separate.
- Do not relabel validation as test, describe a reused test cohort as untouched, or merge separate checkpoint maxima into one result.
- Include material negative comparisons and changed recipes; a thematic research narrative need not reproduce daily troubleshooting.
- Added CBAM gave mixed controlled results; S79 changes more than attention. XAI is qualitative and does not establish clinical validity.
- Preserve frozen configs, class order, checkpoints, original metrics, predictions, hashes and historical evidence.
- No claim of a completed ten-fold evaluation, expert localization validation or clinical deployment is supported.
