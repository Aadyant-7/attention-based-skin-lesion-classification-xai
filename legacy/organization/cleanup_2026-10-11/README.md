# Approved organization cleanup — 11 October 2026

Branch: `structured-research`; starting commit `fb960f9`. Physical project: `E:\Minor Project\Project Development`. The old C: path remains a compatibility junction. Scope: organization and navigation only; no experiments, training, inference or report drafting.

## Applied changes

- Added `reporting/` with evidence navigation, a source/hash manifest and reserved report/paper/presentation folders.
- Archived exact copies of nine old public navigation files before replacing current entry points. See [archives.csv](archives.csv) and [old navigation](../../navigation/cleanup_2026-10-11/).
- Preserved the old private reference index beside its originals, then grouped nine guide-PDF preview PNGs into its `previews/` folder.
- Moved `.cache/prepare_strict.py` unchanged into [inactive maintenance](../../maintenance/README.md). It must not be executed.
- Deleted only 232 regenerable `.pyc` files and one obsolete 328-byte S19 atomic-write residue; removed 23 empty bytecode directories. Recovered 2,533,761 bytes, approximately 2.53 MB.
- Updated active root/research/results/docs navigation and added a question-based study index. Historical scientific proposals, closeouts, freezes and seven historical docs notes remain unchanged.

## Preservation and receipts

[relocations.csv](relocations.csv) records ten moves with original/current paths, size and SHA-256. [deletions.csv](deletions.csv) lists the 233 approved disposable files and their original hashes. Earlier organization receipts were not rewritten.

The full 67,800-file pre-cleanup snapshot is local at `.cache/cleanup/20261011/before_manifest.json`, SHA-256 `1eaf30ad7aa197e23f93912fc14584a5639e3abc235f066181dbeeb9894050be`. Fresh hashes were collected for scientific source, results, all 149 checkpoints, data and cache assets; unchanged environment hashes reuse the verified migration plus identical metadata. Nested Git stores were excluded.

Results, checkpoints, frozen configs/source, all fifteen split files, original notebook, six old Word/PDF reports, private references, external pretrained assets, test guards, numerical failures, logs and recovery states remain at stable paths. Many executable packages compute root from their location or are frozen by path and hash, so moving them for appearance would break reproducibility. Their active/historical roles are now indexed.

[Verification](VERIFICATION.md) and [machine-readable outcome](verification.json) record the completed non-GPU gates. [Current verifier](../../../research/maintenance/README.md) is read-only. The local full snapshot is not duplicated into Git; public ledgers contain provenance, not private reference assets or weights.

## Current navigation

- [Root overview](../../../README.md)
- [Reporting evidence](../../../reporting/evidence/INDEX.md)
- [Research study index](../../../research/STUDY_INDEX.md)
- [Saved result map](../../../results/README.md)

Training remains closed. Curated narrative may group experiments by research question; protocol boundaries, meaningful negative findings and the validation/test distinction must remain visible. Existing XAI is specifically the older S80 five-model package, not exact S83 six-model explanations.
