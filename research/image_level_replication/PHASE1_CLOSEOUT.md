# Phase 1 completed: separate image-level study v1

10 October 2026. User approved Phase 1 only. No training, inference or test scoring was performed. No new model accuracy exists.

## Frozen split

**7,009 training / 1,503 validation / 1,503 test original images**, approximately 70/15/15 as required by the guide. Stratified seven-class, canonical image-ID order, seed 42, one two-stage split. Original image IDs are unique/disjoint; lesion IDs may overlap naturally. No seed search, augmented copies or resampling preceded splitting.

Manifest: `data/splits/image_level_replication/v1/split_assignments.csv`.
SHA-256: `3ce1c7924b52c5e9434b51b1e09bbbda75d438935aab4017a8ceebff8da4fcdf`.
Frozen policy: `research/image_level_replication/protocol_v1.json`.
Class order: akiec, bcc, bkl, df, mel, nv, vasc.

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| akiec | 229 | 49 | 49 |
| bcc | 360 | 77 | 77 |
| bkl | 769 | 165 | 165 |
| df | 80 | 17 | 18 |
| mel | 779 | 167 | 167 |
| nv | 4693 | 1006 | 1006 |
| vasc | 99 | 22 | 21 |


| Pair | Shared lesions | Images in second partition with a lesion in first |
|---|---:|---:|
| train / val | 513 | 536 |
| train / test | 518 | 553 |
| val / test | 138 | 148 |


536 validation images (35.66%) and 553 test images (36.79%) share a lesion with training. These are overlap statistics, not performance results. The full original-image proportions and natural evaluation class counts remain intact.

## Preservation and interpretation

All 617 protected historical file hashes match, including original splits, old test results, reports and frozen S83 method. The master registry's 588 historical rows and bytes are unchanged. All 127 historical checkpoint paths/sizes/mtimes are unchanged. Preparation writes only its new namespace and navigation documentation.

The old test assignments remain unchanged in their original manifest. The new manifest reassigns 1,073 old-test images to training, 224 to validation and retains 206 in the new test. This is approved re-use in a separate post-development study, not a change to the old result. Old project checkpoints/fitted models/probabilities cannot initialize or form an ensemble in this study because their previous training images occur in the new test. Use fresh external pretrained weights.

The prior 94.011976% exploratory validation / 87.558217% S83 audit remains saved and reported. Describe any future new result as **post-development image-level evaluation with lesion overlap**; it is not a pristine first test or lesion-independent performance.

## Saved package and checks

- `research/image_level_replication/PLAN.md`: complete phased plan and GPU approval/launch-and-stop policy.
- `results/image_level_replication/v1/protocol/`: split audit, class/overlap CSVs, PNG/PDF figures, historical cohort reassignment, preservation snapshot, completion and verification records.
- `results/image_level_replication/v1/experiment_registry.csv`: schema only, zero model experiments/results.
- `checkpoints/image_level_replication/v1/`: reserved future location; no checkpoints created.

CPU checks and independent review passed: metadata consistency, all 10,015 filenames present without opening images, exact class order/counts, pairwise image disjointness, deterministic reproduction, frozen hashes, train/validation helper excluding test outputs, preservation and figure/table availability. Preparation is idempotent and refuses a changed/partial freeze.

The preservation verifier allows future new-study registry entries while requiring every historical row to remain identical, and excludes this new study's future checkpoint namespace from the historical inventory. It continues to protect the 127 pre-existing checkpoint files.

## Next phase

Phase 2: verify and prepare one plain EfficientNet-B3 paper-guided adaptation. Input/augmentation/optimizer/balancing details not fully established by the paper must be recorded as assumptions. No training config, GPU start command, achieved score or test-inference runner exists for this study yet. Stop after Phase 1; GPU launch needs the concrete Phase 2 proposal and approval.
