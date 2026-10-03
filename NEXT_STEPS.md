# Research plan: 3–17 October 2026

The two-week target is a planning window, not a promise of a score. Final architecture is open to controlled evidence. No new training or test evaluation was launched in Phase 1.

## Phases and gates

| Phase | Work | Gate / output |
|---|---|---|
| 1 | Audit, preserve legacy, common registry/model interface/artifact logger, dataset counts and saved-result figures | Completed infrastructure audit; history unchanged; test images unopened. |
| 2 | Verify literature rows, dataset/protocol text, research question and comparison recipe | Review table, research gap, registered comparison budget; no inferred literature metrics. |
| 3 | Implement/review common training runner, then sequential transfer-learning comparisons | 4–6 completed comparable run packages; no test selection. |
| 4 | Paired no-attention vs CBAM on 1–2 strongest backbones | Same head/recipe/seed; show class-wise changes and compute cost, including negative results. |
| 5 | One or two hypothesis-led improvements on strongest candidates | Predeclared metric/compute gates; avoid another broad tuning sweep. |
| 6 | Small set of two-model probability ensembles; third member only if justified | Members/weights fixed from development evidence; report inference cost. |
| 7 | Choose final single model or ensemble and freeze methodology | Exact configs, checkpoint hashes, class order, split and preprocessing written down. |
| 8 | One explicit locked-test evaluation | Only after freeze; save full metrics, predictions, matrix and plots. |
| 9 | XAI target-layer verification and representative explanations | Correct/incorrect melanoma, common/minority classes, confidence and true/predicted labels. |
| 10 | Paper tables/figures, results/discussion/limitations and full draft | Guide review; venue/template formatting after venue is known. |

Start literature writing in Phase 2 and methods/setup drafting alongside training; reserve final claims until Phases 7–9. This overlap saves time without inventing results. Phase 9 may use development cases to validate XAI implementation earlier, but final test examples follow Phase 8.

## Shortlist for Phase 3

The installed environment was checked: RTX 4060, 8 GB VRAM; PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128. `research/backbone_shortlist.csv` records exact installed weight-enum names and original ImageNet parameter counts; those are not HAM10000 results.

| Candidate | Role | Compute decision |
|---|---|---|
| EfficientNet-B0 | Small anchor linking to historical work | New no-attention common-head baseline; historical B0+CBAM remains a reference. |
| MobileNetV3-Large | Lightweight alternative | Useful deployment/compute comparator; do not assume it beats B0. |
| ResNet50 | Residual CNN comparator | Distinct family, explicit ImageNet weight version; smaller microbatch if needed. |
| DenseNet121 | Dense-connectivity comparator | Complementary family; memory checkpointing only if documented consistently. |
| ConvNeXt-Tiny | Modern convolutional comparator | Larger compute cost; include only after a short memory/time feasibility check. |
| EfficientNet-B2 | Same-family scale check | Optional sixth; do not add if four/five runs already answer the comparison or deadlines tighten. |

Choose five initial candidates; B2 is optional. These are feasible *candidates*, not verified training-memory fits. No model weights were downloaded to make this recommendation. Official model/weight references: [torchvision models](https://docs.pytorch.org/vision/stable/models.html), [EfficientNet-B0](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html), [ConvNeXt implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/convnext.html). A larger EfficientNetV2-L/ViT-L sweep is not a sensible first use of this GPU or deadline. Domain-specific pretraining remains allowed, with overlap/license caveats.

## Fair-comparison recipe to freeze in Phase 2

- Use the exact existing lesion-disjoint manifest, seven classes, fixed seed and macro-F1 checkpoint selection; report accuracy, macro precision/recall/F1 and class metrics together.
- Start with 224-square ImageNet-normalized input, one documented augmentation recipe, training-only weighted CE, common seven-class GAP/dropout/linear head and no added CBAM. Record native architectural normalization and pretrained recipe differences.
- Use a common effective batch size (proposed 32), 20-epoch cap, five stale-epoch patience, AdamW and documented backbone/head learning rates. Run a short timing/memory preflight per backbone and adjust microbatch/accumulation rather than silently changing effective batch size. BatchNorm microbatch differences remain a limitation; avoid batch size one.
- Log exact weights, hashes, environment, seed, trainability/freeze stages, parameter counts, epochs and wall time. Similar epoch budgets do not imply identical optimization quality; disclose this rather than overclaiming a universal best architecture.
- Implement and record deterministic settings/worker seeds in the new runner; seed 42 alone is insufficient. Explicitly handle already-completed runs and safe resume so post-training bookkeeping never depends on an uninitialized epoch.
- The example config in `research/configs/` is not yet a reviewed training runner. Phase 3 must implement and verify that runner, resume/checkpoint semantics and validation-only loading before launching.
- Historical B0 runs took roughly 10–43 minutes; different backbones/heads/augmentation can take longer. Derive a real remaining budget from preflight timings; do not claim an unmeasured exact ETA.
- If useful, repeat the top paired attention comparison with a second seed. A single seed cannot establish statistical significance.

## Suggested allocation

Days 1–2: Phase 1/2 and protocol freeze. Days 3–6: model comparison. Days 7–9: attention and limited improvements. Days 10–11: ensembles and final selection. Days 12–13: one test evaluation and XAI. Day 14: paper evidence pack and guide review. Reduce optional runs before sacrificing controls or reporting accuracy.

## What is needed from the team

Enough data, split manifests, local checkpoints and dependencies are already available for Phase 1 and planning. No user file is required to complete this phase. Before later final paper formatting, a target journal/conference template would help; before scheduling all runs, the actual daily GPU availability/meeting date would refine the budget. Neither blocks Phase 2. Do not ask the team to recollect material already in the repository.

## Commands available now

```powershell
# Rebuild indexes/tables/figures only; no model execution
.\.venv\Scripts\python.exe -m research.build
# Verify infrastructure only, using saved development evidence/synthetic tensors
.\.venv\Scripts\python.exe -m unittest discover -s research/tests -v
```

No manual command is required to finish Phase 1. Next authorized scope should be Phase 2; do not start legacy training commands or invoke `src.evaluate --confirm-locked-test` now.
