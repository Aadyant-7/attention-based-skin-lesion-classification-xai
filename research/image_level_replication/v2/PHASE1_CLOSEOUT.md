# Revised Phase 1 closeout — V2 K10-inspired protocol

10 October 2026. **Phase 1 completed; no training, inference or new model accuracy.** User relaxed the earlier fixed 70/15/15 requirement before Phase 2. V2 is a new namespace; V1 remains intact.

## Protocol choice

[DermAI 1.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10573070/) associates its B3 97.01% result with K10, but also describes 80/20 elsewhere. We adopt one deterministic stratified image-level ten-fold protocol, not an exact reproduction of unresolved paper details. See [verified/unresolved methods](PAPER_PROTOCOL_REVIEW.md).

All **10,015 originals / 7,470 lesions** enter canonical image-ID ordering. Stratified outer folds use shuffle seed 42; each original is assigned to one outer assessment exactly once. Within each outer development pool, stratified inner validation uses 10% of that pool, with seed 42 plus fold ID.

| Outer folds | Actual fitting | Inner validation | Outer assessment | Outer development pool |
|---|---:|---:|---:|---:|
| 0–4, each | 8,111 | 902 | 1,002 | 9,013 |
| 5–9, each | 8,112 | 902 | 1,001 | 9,014 |

Effective fitting/validation/assessment is approximately **81/9/10**, not 90% fitting. A later fresh refit on the outer 90% can use an inner-selected fixed epoch budget; this adds training and needs a separate proposal/approval. No such refit or ten-fold training queue exists now.

## Overlap and interpretation

All original image IDs are disjoint within each fold. Different photographs of the same lesion may cross partitions naturally; overlap was measured, not optimized. Fold 0:

| Pair | Shared lesions | Images in second partition sharing the first partition's lesion |
|---|---:|---:|
| Training / inner validation | 339 | 353 / 902 |
| Training / outer assessment | 375 | 389 / 1,002 |
| Inner validation / outer assessment | 62 | 64 / 1,002 |

Per-class counts and all fold overlaps are saved. Split originals before augmentation/oversampling; apply those operations to training only. Inner validation and assessment retain natural class proportions.

Future scores must be labelled **post-development internal image-level K10-inspired evaluation with lesion overlap**. Prior outcomes are known; these partitions cannot recover a pristine external estimate. Earlier strict and S83 results remain separate.

## Checks and preservation

CPU metadata/filename verification passed: canonical reconstruction of all ten folds, frozen hashes, exact class mappings, presence of every class in every role, each original assessed exactly once, no original-image overlap, all original filenames present, and inner-development helper excluding assessment IDs. Independent read-only reconstruction also passed, including V1 comparison against Git HEAD. Repeated preparation left all V2 artifacts unchanged; in-memory changed-hash and invalid-fold checks were rejected. Both PNG protocol figures were visually checked.

Preservation checks passed for **636 historical/V1 files**, **588 historical master-registry rows**, and the unchanged path/size/modification-time inventory of **127 historical checkpoint files**. Historical result hashes are preserved; checkpoint inventory is not a claim of newly hashing checkpoint contents. The V2 registry is header-only, with zero model experiments. No image was decoded or scored and no GPU was used.

## Saved artifacts and next boundary

- Policy/hash ledger: `research/image_level_replication/v2/protocol.json`.
- Assignments: `data/splits/image_level_replication/v2/outer_folds.csv`, `fold_00.csv` through `fold_09.csv`.
- Counts/overlap/audit/verification/preservation/completion receipts: `results/image_level_replication/v2/protocol/`.
- Readable PNG/PDF protocol figures: sibling `figures/`.
- New empty registry: `results/image_level_replication/v2/experiment_registry.csv`.
- Future checkpoint namespace: `checkpoints/image_level_replication/v2/`; none created.

Repeated `--prepare` verifies existing artifacts without rewriting them. `--verify` is CPU-only. Phase 2 must prepare one fresh plain-B3 adaptation and a concrete resource/start/monitor proposal for **fold 0 inner validation only**. No GPU launch is approved by this closeout.

Full K10 costs roughly ten separate runs per backbone. Each assessment image must use only its own fold's constituent models; averaging all ten fold models would include models trained on that image. Report pooled OOF metrics, every fold and mean/std only once all ten are complete; otherwise report partial evaluation honestly. No assessment scores may choose checkpoints, weights or later methods.
