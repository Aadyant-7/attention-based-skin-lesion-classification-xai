# Reporting workspace

Prepare the final university report, research paper and presentation from saved evidence here.

- [Evidence index](evidence/INDEX.md): current outcomes, original figures and protocol/method boundaries.
- [Final product handoff](FINAL_PRODUCT_HANDOFF.md): retained six-model framework, exact result/XAI paths and practical opening instructions.
- [Source manifest](evidence/source_manifest.csv): original file paths and current SHA-256 hashes. Metrics are fractions, populated only when present in that actual source file.
- [Report](report/README.md), [paper](paper/README.md), [presentation](presentation/README.md): writing and presentation areas; current Word/PDF drafts are local-only.

Original experiments remain in `results/`, local checkpoints in `checkpoints/`, protocols in `data/splits/`, and decisions/literature in `research/`. Checkpoints are referenced through existing manifests, not copied. The full editable report and compact two-column paper drafts have been produced and visually checked locally under `reporting/report/draft_v2/`; these private drafts are not included in GitHub. No new model training or evaluation was performed for writing.

Current completed XAI explains the exact S83 six-model method under `results/final_s83_xai/v1/`; verification/manifest receipts passed with zero changed labels. Earlier S80 five-model XAI remains historical. Scientific performance is unchanged: 94.011976% exploratory validation and 87.558217% repeated post-development test audit. XAI is explanatory evidence, not an accuracy improvement.

Later writing can follow research questions and justified improvements rather than daily chronology. Summarize neutral/unsuccessful studies briefly or in an appendix, retain material comparisons, and distinguish exploratory validation, strict validation and test/post-development audits. The full registry remains the audit trail.
