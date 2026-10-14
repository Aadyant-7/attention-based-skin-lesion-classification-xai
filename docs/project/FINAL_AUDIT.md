# Final repository audit and consolidation

**Date:** 14 October 2026. **Outcome:** source/evidence preservation, local checks and the tracked-only download check passed. This is an organization/publication audit, not another model experiment.

## Consolidated Git state

The published active/default branch is `main`. Before finalization, main was an ancestor of accuracy-exploration, which was an ancestor of structured-research. Structured-research contained 101 commits absent from main and 93 absent from accuracy-exploration; neither older branch contained unique work missing from it. Main was advanced without squashing or rewriting that history.

Historical tips are preserved by annotated tags:

| Tag | Original commit |
|---|---|
| `archive/main-before-consolidation-2026-10-14` | `ac 868d 4e 72c 51f 5ea 9fc 55bcbb 37cf 9b 5e 8ecee 3` |
| `archive/accuracy-exploration-2026-10-14` | `8abbc 29a 302357c 2e 1df 07fcd 6fb 5fd 6f 71fde 0c` |
| `archive/structured-research-2026-10-14` | `ba 59354` |

`strict-baseline-snapshot` remains preserved. The two redundant development branches are retired after main and the archive tags are verified on GitHub. Earlier branch names in dated historical reports remain unchanged.

## Organization and documentation

- Root Markdown is the main README. Current project status and human/assistant handoff moved to [STATUS.md](STATUS.md); the move retains Git history.
- Added [SETUP.md](../SETUP.md) for cloning, saved-result presentation, optional code dependencies and local-asset requirements. Main README now indexes exact result and XAI paths.
- Research plans/handoffs remain beside their studies; no generic agent-instruction file was found at the root. Original research code/results/checkpoint namespaces remain stable because frozen receipts and imports depend on them.
- Updated current navigation to main and the relocated status. Replaced two misleading clone links to an ignored private reference and original SciSpace workbook with explicit local-only paths and tracked evidence navigation.
- Local full-report/paper drafts are documented as completed review drafts, still private/ignored. Guide approval, actual contributions and signatures remain pending.
- Declared directly imported research dependencies that previously relied on transitive installation. Existing environment passed `pip check`; a clean network dependency reinstall was not performed. Viewing evidence and the clone audit require none of these ML packages.

## Real download problems found and fixed

1. **Historical registry references:**14 references pointed into ignored old run directories. The clone audit now uses their already-recorded portable copies: 5 configs, 3 histories, 3 metric records and 3 exact confusion-matrix aliases. All 15 portable mirror files are SHA-256 authenticated. Original registry rows and scientific files were not rewritten.
2. **Git newline conversion:**26 authenticated source/result files had different newline storage in Git than the frozen local bytes. Added explicit byte-preserving `.gitattributes` rules and staged those original authenticated bytes. Every difference was verified to be newline-only; no numbers, model logic, checkpoint selectors or original local bytes changed. Existing SHA-256 receipts now survive the tracked-only export.

## Saved scientific evidence

| Check | Result |
|---|---|
| Master registry | 600 unique experiment IDs; rows include diagnostics/fusions and are not a trained-model count |
| Full local registry navigation | 1, 826 references resolve |
| Clone registry navigation | 1, 700 scientific references resolve, including 14 authenticated portable fallbacks;121 checkpoint and 5 cache references separately labelled local-only |
| Reporting sources | 487 local hashes pass; clone verifies 486 file hashes plus the original workbook digest against tracked SciSpace provenance |
| Frozen S83 |Freeze/split, selected predictions, all six selected checkpoint binaries, six source prediction files and five FP32 cache hashes pass locally |
| Exact S83 XAI | 501 output hashes, 8 frozen source files and all 495 protected historical hashes pass locally; clone hashes 491 historical files and separately records four intentionally ignored log receipts |
| Member curves |All 42 expected tracked history/accuracy/loss/macro-F1 PNG/PDF files exist across six retained members |
| XAI presentation | 12 predefined validation cases, 72 branch maps, saved gallery/atlas;39 gallery resources resolve; existing numerical/visual receipts pass |
| Source/runtime | 201 Python files parse;15 reviewed CPU imports pass; CUDA remains uninitialized; R202/R203 signatures unchanged |
| Git |Object connectivity check and staged whitespace checks pass; no force push or scientific history rewrite |

Final S83 remains **94.011976% exploratory validation /0.895200837 macro-F1** and **87.558217% recorded post-development test audit /0.807031200 macro-F1**. The latter cohort was previously evaluated. Original strict S31 and separate R202/R203 results retain their own evaluation roles. Frozen manifests, selected epochs, weights and class order remain unchanged.

## Download verification and repeatable commands

An exported Git tree was extracted into a new directory without `.git`, `.venv`, caches, checkpoints, private references or report drafts. The new audit passed there with Python **site-packages disabled (`-S`)** and bytecode disabled (`-B`). This validates the actual tracked source/evidence package, rather than relying on files that happen to exist on the project machine. This audit does not claim an independently reinstalled GPU environment or a measured clinical deployment.

From any clone:

```powershell
python -B -m research.maintenance.verify_repository
```

The audit checks 201 Python files, 600 registry IDs, frozen evidence, source hashes, active links and gallery resources. It uses only the standard library and writes nothing by default. In the final checked navigation, 20 active Markdown pages and 303 local links resolved. If navigation is extended later, the live output is authoritative for its count.

From the original full local workspace:

```powershell
& '.\.venv\Scripts\python.exe' -B -m research.maintenance.verify_organization --portable
& '.\.venv\Scripts\python.exe' -B -m pip check
```

The older `--full-hash` cleanup baseline describes 11 October and is not a current-state gate after later authorized additions. No historical receipt was rewritten to conceal changes.

## Publication scope and remaining work

Tracked results, figures, CSV/JSON records and XAI are present for presentation. Raw images, checkpoint binaries, pretrained caches, original SciSpace XLSX, private guide references, live logs and local editing drafts remain local and preserved. Saved results can be presented without them. Original metrics and XAI receipts were not regenerated; only two current navigation hashes in the reporting source index were refreshed after their documented text edits.

No training, model/checkpoint loading, image decoding, new ensemble search or test inference/scoring occurred. Private draft folders and credential-shaped paths were excluded from staging; the reviewed source/config/text token scan found no credential signatures. This is a scoped publication check, not a claim of a formal security assessment.

Remaining work is group/guide review, report/paper revisions, real contribution/signature completion, presentation and submission. [Current status](STATUS.md) provides the handoff. The clone instructions do not schedule another experiment or held-out evaluation.
