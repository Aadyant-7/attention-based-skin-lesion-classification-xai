# Active paper-guided study V2: K10-inspired image-level evaluation

User relaxed the earlier 70/15/15 requirement on 10 October 2026. V2 replaces V1 as the active plan; all V1 manifests, policy, code, audits and figures remain preserved. No model training or accuracy result exists for either new version.

**Phase 2 preparation:** [B3 proposal/commands/artifacts](PHASE2_PROPOSAL.md) and [recipe evidence/assumptions](RECIPE_DECISIONS.md). An isolated fold-0 runner and CPU preflight are implemented; launch requires a verified freeze and separate GPU approval. Preparation checks are recorded under `results/image_level_replication/v2/preparation/`. No trained run or outer-assessment inference exists.

## Why this protocol

[DermAI 1.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10573070/) associates its reported B3 97.01% with K10. It also describes 80/20 elsewhere; exact assignments and augmentation lineage are unavailable. We choose one declared stratified ten-fold image-level protocol, with separate inner validation. This is a documented adaptation, not a verified exact reproduction or a prediction of 97% performance. See [paper-method review](PAPER_PROTOCOL_REVIEW.md).

| Role, per outer fold | Originals | Approximate fraction |
|---|---:|---:|
| Actual fitting | 8,111–8,112 | 81% |
| Inner validation | 902 | 9% |
| Outer assessment | 1,001–1,002 | 10% |

The outer development pool is 90%; reserving 10% of that pool for inner validation leaves 81% for actual fitting. A later **fresh refit on the outer 90%** can use an inner-selected fixed epoch budget, followed by one outer assessment; that additional training pass requires a separate proposal/approval. It is not implemented or launched in Phase 1.

Every original image is assigned to exactly one outer assessment fold. Different photographs of one lesion may appear across roles. Original image IDs within a fold are disjoint. No split/seed selection is based on model scores, and no augmented copies are assigned across roles.

## Locations

- `data/splits/image_level_replication/v2/outer_folds.csv`: one outer-fold assignment per original.
- `data/splits/image_level_replication/v2/fold_00.csv` through `fold_09.csv`: explicit inner train/validation/outer-assessment roles.
- `research/image_level_replication/v2/protocol.json`: frozen algorithm, hashes, class order, selection/aggregation rules.
- `research/image_level_replication/v2/PLAN.md`: current phased sequence and compute policy.
- `results/image_level_replication/v2/protocol/`: counts, overlap, preservation, verification and PNG/PDF figures.
- `results/image_level_replication/v2/experiment_registry.csv`: new run registry; header only until a real run starts.
- `checkpoints/image_level_replication/v2/`: future local checkpoints; none created by Phase 1.

## CPU-only commands

```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
.\.venv\Scripts\python.exe -B -m research.image_level_replication.v2.prepare_protocol --prepare
.\.venv\Scripts\python.exe -B -m research.image_level_replication.v2.prepare_protocol --verify
```

Preparation freezes all folds once; repeated preparation verifies without rewriting them. These commands read metadata/filenames, never decode images, import torch, train, run inference or calculate model accuracy. `load_development(fold=0)` returns only the chosen fold's inner training/validation metadata after frozen checks. This helper is not a training runner.

## Evaluation and budget

Start future feasibility/development on **predeclared fold 0, inner validation only**. Outer assessment does not select checkpoints, epoch budgets, weights, architectures or later modifications.

Completed K10 requires ten independent same-method fold runs per backbone, fresh external pretrained initialization in every fold, and one out-of-fold prediction per original. Same-fold ensemble components can fuse an image's probabilities; averaging all ten fold models would include models trained on that image and is prohibited for OOF evaluation. Report pooled OOF metrics, every fold and mean/std; never report a best fold as completed K10.

All ten folds are prepared, not scheduled. A full K10 study costs approximately ten training runs per backbone; optional fresh outer-90% refits add runs. The Phase 2 proposal specifies the first bounded inner-fold B3 run and its resource/monitoring details. The prepared allowance is 50 epochs with a 25-epoch minimum and meaningful stopping, superseding the earlier provisional 30-epoch plan. Launch-and-stop remains in effect.

Label results **post-development internal image-level K10-inspired evaluation with lesion overlap**. Previous test outcomes were known. Old S83 94.011976% validation / 87.558217% audit and all strict results remain reported separately.
