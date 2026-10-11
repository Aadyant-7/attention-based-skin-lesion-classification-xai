# Final local and Git readiness audit

Verified 11 October 2026 on `structured-research`, after cleanup commit `42cfbc4`. **Organization is finalized and report writing can begin.** No scientific outputs, checkpoints, configurations or datasets were changed; no training or inference ran.

## Preservation and Git

- All 67,547 preserved original files retain the metadata checked at the full SHA-256 cleanup gate; no unexpected changes. That gate freshly authenticated scientific source, results, data/cache assets and all 149 checkpoints.
- All 286 curated evidence hashes still match. Of these, 285 files are tracked; the original SciSpace workbook is intentionally local-only.
- Git history connectivity passes. The local branch and actual GitHub branch matched at `42cfbc4`; main and accuracy-exploration remain separate preserved branches.
- The E: project is the actual working tree. The former C: path remains a compatibility junction.
- Weights, raw data, environment/cache assets and private references remain intentionally outside Git. A GitHub clone is not a full copy of these local assets.

## Saved media and result consistency

| Check | Outcome |
|---|---|
| All saved media under results/docs/legacy | 1,204 PNG/JPEG images decode and verify; 1,082 PDFs parse, have valid pages and readable content streams; zero failures. |
| Representative visual review | Rendered S83 normalized matrix, R203 accuracy curve and original strict-test class plot have readable labels and no clipping/overlap. This is a sample layout review, not page-by-page visual review of every PDF. |
| Main result packages | 32 selected packages; 32 metrics JSONs and 32 prediction CSVs, 48,096 prediction rows checked. |
| Probabilities | 16 separate probability CSVs plus probability columns embedded in prediction CSVs; finite normalized values and predicted argmax agree. |
| Class scores | 37 CSVs agree with saved global metrics and predictions. |
| Confusion matrices | 32 raw + 32 normalized CSVs; raw totals equal 1,503 images and normalized rows sum to 1. Physical CSV copies do not imply distinct experiments. |
| Trained model curves | Ten requested backbone packages have saved accuracy, loss and macro-F1 curves in PNG/PDF. |

The backbone audit covers S02, S03, S05, S06, S10, S15, S18, S19, S79 and R203, including available accuracy/F1/final selectors. S15's 17 contiguous epochs reflect its documented patience stop, not missing history; the other seven early backbones have 20 epochs, S79 has 40 and R203 has 30. Failed or administratively stopped studies remain labelled as such; no completed score is invented for them.

## References and explainability

- Guide PDF, two screenshots, nine grouped previews and original reference papers are preserved; private files remain excluded from Git.
- Original SciSpace workbook matches its extraction provenance; integrated screening/review and verified-source bibliography are available.
- Original 41-cell Colab notebook and six historical Word/PDF reports match preservation hashes. All three Word archives pass ZIP integrity checks.
- Existing S80 XAI retains ten validation cases, 50 model explanations, 130 PNGs, 20 PDFs and 120 finite NumPy arrays, with correct five-model/0.2-weight provenance.

## Remaining reporting work, not organization failures

1. **Exact S83 six-model XAI is not implemented yet.** Current Grad-CAM panels explain the earlier S80 five-model method. Keep those captions explicit; create matching S83 explanations separately if the final report needs them.
2. Select and format a current report comparison table from the saved results. Earlier model-vs-accuracy tables are stage snapshots, not complete current leaderboards.
3. Correct the pre-existing bibliography author-name encoding during citation preparation; verified source metadata preserves the correct name. Original reference bytes were not rewritten during this audit.
4. Apply the guide's format, team details/contributions, captions, bibliography and layout during drafting. Those deliverables are not yet written in the reserved reporting folders.

Highest retained result remains **94.011976% exploratory validation / 0.895200837 macro-F1**. Strict validation, the original strict test and subsequent test audits keep their distinct protocol labels. Organization completeness does not convert validation into independent test performance.

[Evidence index](INDEX.md) · [Machine-readable audit](FINAL_READINESS_AUDIT.json) · [Earlier full cleanup verification](../../legacy/organization/cleanup_2026-10-11/VERIFICATION.md)
