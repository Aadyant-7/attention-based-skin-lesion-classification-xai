# Literature review evidence workflow (Phase 2)

**Focused Phase-2 review completed:** open [review table](review_table.md), [full CSV](review.csv), [evidence notes/research gap](evidence_notes.md) and [citations](references.bib). `sources.json` contains curated verified fields; `research.phase2.prepare` validates and exports them. This is not an exhaustive systematic review and does not claim to reproduce literature metrics. Missing fields stay explicitly unverified.

`review_template.csv` supplies the guide's requested fields. Start by transferring evidence from `historical_evidence_index.csv` and the linked original audits. Do not fill missing details from model conventions. Use `Not reported` for absent source statements and `Not verified` for uninspected claims. Record source URL, page/table/section, date checked, and which fields were independently verified.

The source-specific research tables already exist in `docs/accuracy_exploration_research.md`, `docs/high_accuracy_claims_audit.md`, `docs/base_paper_protocol_audit.md`, and `docs/panderm_probe.md`. These are historical evidence, not a newly completed comprehensive literature review. Preserve distinctions among seven-class and reduced-class tasks, image-level and lesion-disjoint splits, internal validation and final tests, natural and balanced populations. No new paper accuracy is inferred in Phase 1.

The focused set covers dataset/protocol context, transfer-learning comparison, attention/CBAM, balancing, fusion, domain pretraining and XAI. Expand only for a specific paper argument or experiment question. Venue-specific citation formatting can follow once a publication venue is chosen.
