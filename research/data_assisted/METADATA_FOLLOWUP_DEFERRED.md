# Metadata follow-up: deferred

**Follow-up completed2026-10-06:** S68/S69 tested fixed train-only metadata adjustments with the retained S53 ensemble. Age-only reached93.4797% /0.882840 macro-F1; age+sex+location reached93.0805% /0.877659, both below image-only93.6128% /0.886857. Both material gates failed and melanoma recall declined. Keep image-only; no metadata-strength/bin search or multimodal GPU training follows. Full report: `research/data_assisted/S68_S69_METADATA_CLOSEOUT.md`. The earlier deferred note and historical scores below are preserved.

The user requested a later revisit of metadata with the current ensemble. No new metadata classifier or fusion was run during external-data clearance.

Earlier evidence is saved under `results/phase5_multimodal_probe/`, especially `feasibility_conclusion.md`, `metadata_only_results.json`, `best_multimodal_classifier.json`, and `best_fusion_accuracy.json`.

| Earlier method | Validation accuracy | Macro-F1 |
|---|---:|---:|
| Metadata only: age, sex, localization |27.8776%|0.185897|
| Image embeddings only |83.8323%|0.734722|
| Image embeddings + age |84.0985%|0.738495|
| Small image/metadata MLP |84.2315%|0.745492|
| Best metadata-aware fusion by accuracy |84.6307%|0.763740|
| Earlier best image-only reference |85.7618%|0.771810|

The earlier metadata study failed its +1.5 percentage-point improvement gate. Age was the most useful metadata field, but there was no evidence to justify a full multimodal CNN then. These scores are from an earlier lesion-disjoint development cohort and cannot be directly compared to today's 93% exploratory ensemble.

Later question: does clinically available age/sex/localization add complementary information to the fixed current ensemble? Use matched image identities, fit imputers/encoders and any fusion only on training folds, and keep the original locked test closed. Diagnosis-confirmation type, image/lesion identifiers and other diagnosis-derived fields must not become predictive inputs. Record class-wise and melanoma effects, not only accuracy. Revisit after the current data preparation step; benefit is not assumed.
