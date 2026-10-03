# Two-stage evaluation — Phase 3 amendment

**After S04:** [Fixed equal fusion](S04_CLOSEOUT.md) achieved **88.56% exploratory accuracy/.8057 macro-F1**, improving over [B0](S03_CLOSEOUT.md)86.36%/.7707 and [MobileNet](S02_CLOSEOUT.md)86.03%/.7861. Retain this bounded-fusion reference; next recommendation is [matched B0+CBAM](S05_PROPOSAL.md), prepared only. Strict results remain separate. Pre-S02 rationale below is historical, not authorization for another run.

**GPU launch policy:** every future GPU run needs its own reviewed proposal and approval. Once launched independently and log/initial checkpoints confirmed, stop immediately. No epoch polling, background monitoring or completion wait; user returns to request analysis.

User decision 3 October 2026: prioritize **image-level exploratory discovery**, then **fresh strict confirmation of only the strongest methods**. This supersedes screening every backbone on the strict split. Phase2 and S01 plans/configs remain historical evidence.

| Stage | Existing manifest | Train / validation / locked test |
|---|---|---:|
| Exploratory screening | `data/splits/exploratory/image_level_dev_v1.csv` |7,009 /1,503 /1,503 |
| Strict confirmation | `data/splits/split_assignments.csv` |7,009 /1,503 /1,503 |

Actual proportions **69.9850%/15.0075%/15.0075%**, approximately 70/15/15. Exploratory development was stratified at seed 42 within the 8,512 original non-test images. Both retain the same 10,015 identities/labels and identical original 1,503 test assignments. **No new split is created.**

Exploratory hash `75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb`; strict hash `db1ce9f8b83f176001dd89fd332fb1668c7d2ba5f58d77157d293a09e2c2e696`. Metadata-only audit: `results/datasets/two_stage_protocol_audit.json` and `two_stage_class_counts.csv`, reproduced by `python -m research.phase3.audit_protocols`.

Exploratory train/validation share563 lesions;596 validation images have a training lesion. Image IDs do not overlap. Test lesions remain disjoint from both development partitions. Label scores **“exploratory image-level development screening with retained lesion-disjoint locked test.”** Literature-style proportions do not establish direct comparability to every paper's cohort/balancing/metric/selection. Neither protocol proves patient independence.

## Discovery rules

`recipe_screening_v1.json` retains S01's224 RGB/ImageNet normalization, horizontal flip, training-only sqrt class weighting, AdamW LR groups, full fine-tuning, batch 16/accumulation2, AMP/determinism and 20-epoch cap. **Accuracy-focused amendment:** select earliest maximum validation accuracy, schedule LR/early-stop after five stale accuracy epochs, independently save the earliest maximum-macro-F1 state/probabilities. `best.pt` is accuracy-selected; `best_macro_f1.pt` is the balanced alternative. S01 illustrates why both should be preserved.

Rank selected accuracy, then macro-F1, then measured cost. Always show macro precision/recall/F1, balanced accuracy, melanoma recall and class supports. Shortlist the accuracy leader and a balanced leader if distinct. Never put accuracy and F1 from different epochs into one row. Secondary checkpoint comparisons identify their selection/epoch explicitly. No arbitrary majority-accuracy score replaces the class-wise review.

**Next: MobileNetV3-Large**, a new lightweight family linked to the base-paper comparison. Its2.979M common-head parameters versus S01's4.017M make it a reasonable bounded diversity experiment. The [original MobileNetV3 paper](https://arxiv.org/abs/1905.02244v5) describes hardware-aware mobile-CPU design; this does not guarantee GPU speed or HAM10000 gains. Expanded literature supports transfer learning, attention and limited fusion; extracted high scores are not promises.

S02 changes architecture, development partition and selection criterion relative to S01; it cannot isolate a backbone advantage or split effect over S01. Historical exploratory B0+CBAM86.23% and ensemble90.75% remain context, with different recipes. If MobileNet is promising, propose a matched exploratory CBAM pair or a common-recipe B0 bridge based on errors/cost. If weak, diagnose fit/convergence and choose one justified alternative family instead of launching the entire list. Larger CNNs, scaling, segmentation/GANs/transformers are options, not an automatic queue.

After controls: one augmentation/balancing change at a time, motivated by errors/curves; resolution/multiscale only if lost detail is plausible; fusion only with aligned complementary predictions. Start fixed equal two-model probabilities, not hundreds of weight/prior searches. Training-only resampling/augmentation; natural validation/test support. Register recipe amendments and matched controls. Image-level splitting does not shorten an epoch by itself; savings come from fewer strict reruns and bounded candidates.

## Strict confirmation and final gate

Normally confirm one accuracy leader and one balanced alternative if different. Predeclare configs, initialize from documented external pretrained weights, and train on **strict training images only** using the selected methodology. Do not reuse exploratory trained weights: **1,243 strict-validation images are already in exploratory training**. Direct exploratory-checkpoint evaluation on strict validation is contaminated. Resume config identity is enforced; no arbitrary warm-start path exists.

Carry declared selection/recipe into confirmation. Any strict adaptation is documented with the required matched control. Repeat a leading attention/control pair at a second seed if affordable. Strict validation has already been used in history/S01; it is not a new blinded test and retains selection bias. Interpret a strict drop as a generalization diagnostic; never conceal it.

Keep separate exploratory, strict and legacy tables. Export comparisons by **protocol + recipe + seed + selection criterion**; registry uses `era=structured`, `phase=exploratory_screening` and explicit validation source. Save full state, both winners, config/environment/pretraining hashes, logs/progress/history, metrics, probabilities, class scores, matrices and curves, including negative outcomes and cost. Grad-CAM follows target-layer verification and representative labelled cases.

Freeze final methodology/checkpoints/ensemble members and weights before one predeclared final locked-test evaluation session; test never selects models. Report both development protocols separately and identify exact training/selection protocol for final test results. No guaranteed90/93% or clinical-effect claim is made. No new GPU work or test inference occurred during this amendment.
