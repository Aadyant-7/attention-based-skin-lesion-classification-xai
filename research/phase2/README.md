# Phase 2 handoff — 3 October 2026

**Complete:** local navigation cleanup, focused primary-source review, dataset/protocol verification and fixed Phase-3 comparison plan. **No GPU run was launched.**

Read in this order:

1. [Folder map](../FOLDER_MAP.md): active vs historical vs local assets.
2. [Literature review](../literature/review_table.md), [CSV](../literature/review.csv), [source notes](../literature/evidence_notes.md): field-level verification, not an exhaustive systematic review.
3. [Dataset/protocol](DATASET_PROTOCOL.md): exact cohort, class counts, limits and test lock.
4. [Experimental setup](EXPERIMENTAL_SETUP.md): fixed settings, questions, selection and amendment rules.
5. [GPU run handoff](GPU_RUN_HANDOFF.md): mandatory announcement, launch/monitor contract and saved artifacts.
6. [Next steps](../../NEXT_STEPS.md): Phase 3 implementation gates before training.

## Decisions

- Five initial ImageNet CNNs: B0, MobileNetV3-Large, ResNet50, DenseNet121, ConvNeXt-Tiny. B2 remains optional; PanDerm remains a historical/domain-pretraining reference, outside this controlled ImageNet comparison.
- Primary ranking uses validation macro-F1, with accuracy and class-wise trade-offs always shown. No target accuracy is promised.
- First compare **without added CBAM**; native SE in B0/MobileNet remains. Later add identical final-feature CBAM to selected models with matched controls.
- Reuse the frozen lesion-disjoint manifest. Do not spend the two-week budget on new exploratory split searches.
- Recipe v1 configs are registered as **planned**, outside the result registry until a real run starts. No placeholder score/checkpoint is created.
- GPU timings/memory fit, the runner, resume and XAI target-layer adaptation are still Phase-3/later work. A preflight is also a GPU run and requires advance explanation.

## Organization record

The root README is now a short active entry point. Two old root handoff guides moved unchanged into `legacy/navigation/`, with hashes and a README snapshot. Historical executable/data/checkpoint/result paths stay intact (reports/notebooks were subsequently archived with a relocation map); the immutable 247-file inventory still verifies. New comparison/XAI locations are explicitly reserved. No historical experiment, report or raw file was deleted.

## Remaining limitations

Literature unknowns remain marked; published scores are not independently reproduced. The base paper's test support and split wording limit comparability. Patient-level grouping cannot be established from these lesion IDs. Historical validation has been repeatedly searched; it is not an unbiased final performance estimate. PanDerm pretraining overlap with downstream images is unresolved. Single-seed comparison cannot establish significance. Final-paper venue and actual GPU availability can refine scheduling later without blocking the current plan.

No user upload is required now. Next: implement/review the shared runner on CPU, then present a concrete GPU preflight proposal. Stop here before training.
