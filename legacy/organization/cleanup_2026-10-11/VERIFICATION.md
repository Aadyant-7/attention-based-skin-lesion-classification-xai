# Cleanup verification — passed

Verified 11 October 2026, after the approved cleanup. Scientific source/configs, registries, saved outputs and checkpoints remain unchanged. No experiments, training, inference or image decoding were performed.

| Gate | Actual outcome |
|---|---|
| Fresh pre-cleanup baseline | 67,800 files, excluding nested Git stores; digest recorded in the cleanup README. |
| Preserved files | 67,547 original files retained at unchanged paths with identical size/mtime. |
| Post-cleanup SHA-256 checks | 19,486 retained non-environment files match the fresh baseline, including all 149 checkpoint files, datasets, results and shared source. |
| Environment | 48,061 files retain baseline metadata; original hashes derive from the verified migration. No dependency changes. |
| Exact navigation archives | Ten verified byte-for-byte: nine public navigation copies and one private reference-index copy. |
| Approved physical moves | Ten verified unchanged: nine previews and the inactive source generator. |
| Approved disposable deletions | 233 exact files; 232 bytecode and one obsolete atomic-write residue. |
| Frozen signatures | R202 and R203 unchanged, including source/config/split/weight/library hashes. |
| Python | 189 research/src/scripts files parse; the archived generator parses but was not executed. |
| CPU imports | Fifteen representative shared/runtime modules import successfully. CUDA is not initialized. |
| Master registry | 593 rows; all 1,780 artifact/path references resolve, including JSON checkpoint lists and stored selectors. |
| Reporting manifest | All 286 original file hashes match. |
| Active navigation | All 257 local links checked by the combined gate resolve. |
| Compatibility path | Original C: directory remains a junction to the physical E: project. |

See [machine-readable outcome](verification.json), [move ledger](relocations.csv), [delete ledger](deletions.csv), [archive ledger](archives.csv) and [cleanup record](README.md).

The local full snapshot stays in `.cache/cleanup/20261011/`, not Git. The [read-only verifier](../../../research/maintenance/README.md) can recheck it with the existing environment. Archived bytes are protected by Git attributes; private originals, raw data, cache assets and weights are not published by this cleanup.

This gate confirms organization/path/runtime preservation; it does not constitute new model evaluation. Highest retained accuracy remains 94.011976% exploratory validation, and the distinct recorded evaluation results are unchanged.
