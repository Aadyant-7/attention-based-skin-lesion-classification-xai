# Active paper-guided study V2: K10-inspired image-level evaluation

User relaxed the earlier 70/15/15 requirement on 10 October 2026. V2 replaces V1 as the active plan; all V1 manifests, policy, code, audits and figures remain preserved. R201 B3 now has real fold-00 inner-validation results. V1 remains preparation only; no outer-assessment inference has been performed in V2.

**Current direction:** R201 stopped cleanly after epoch 28; its [closeout](R201_CLOSEOUT.md), checkpoints and results are preserved. The active next experiment is fresh **ConvNeXt-V2 Base**, prepared on unchanged V2 fold 00; see [R202 Phase 1 proposal](R202_PHASE1_PROPOSAL.md). CPU preparation verifies official weights, full fine-tuning, weighted accumulation, EMA and recovery. GPU feasibility and training await separate approval. No ConvNeXt run has launched on V2. The earlier [Tiny + CBAM proposal](CONVNEXT_NEXT_PROPOSAL.md), [B3 proposal](PHASE2_PROPOSAL.md), [recipe assumptions](RECIPE_DECISIONS.md), config and launch freeze remain historical preparation evidence.

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
- `results/image_level_replication/v2/experiment_registry.csv`: active V2 run registry, including R201.
- `checkpoints/image_level_replication/v2/`: local V2 checkpoints, including preserved R201 best/latest states and its administrative final checkpoint after closeout.

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

All ten folds are prepared, not scheduled. A full K10 study costs approximately ten training runs per backbone; optional fresh outer-90% refits add runs. The original B3 allowance was 50 epochs with a 25-epoch minimum and meaningful stopping; its user-requested cancellation is recorded separately from that frozen policy. The active Base proposal specifies 50 planned epochs, two head-only warmup epochs, stage-wise AdamW/cosine scheduling, train-only square-root class weights and separately evaluated raw/EMA weights. Meaningful counters are observational; an explicit manual stop finishes and commits the current epoch. The earlier unlaunched Tiny proposal's 40-epoch cap remains historical. Launch-and-stop remains in effect.

CPU preparation and status:

```powershell
.\.venv\Scripts\python.exe -B -m research.image_level_replication.v2.preflight_convnext_base
.\.venv\Scripts\python.exe -B -m research.image_level_replication.v2.train_convnext_base --check
```

The preflight uses two disposable CPU optimizer updates to verify correctness, then discards them. It computes no project accuracy and opens no outer-assessment images. Its report and launch freeze are under `results/image_level_replication/v2/preparation/`. See the R202 proposal for the later approved GPU memory check, training, monitoring, recovery and stop commands.

Label results **post-development internal image-level K10-inspired evaluation with lesion overlap**. Previous test outcomes were known. Old S83 94.011976% validation / 87.558217% audit and all strict results remain reported separately.
