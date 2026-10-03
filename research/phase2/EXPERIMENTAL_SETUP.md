# Fixed controlled comparison — recipe v1

Registered 3 October 2026; **planned, not launched**. This is a constrained common-recipe transfer-learning comparison, not a claim that each architecture has been optimally tuned. Configs and `comparison_plan.csv` are under `research/configs/backbone_comparison/`.

## Research questions and bounded order

| Stage | Question | Runs / rule |
|---|---|---|
| Phase 3 | Which ImageNet backbone gives the strongest balanced validation performance at our budget? | B0 → MobileNetV3-Large → ResNet50 → DenseNet121 → ConvNeXt-Tiny; one fixed recipe/seed. B2 optional only after five feasible runs and remaining budget review. |
| Phase 4 | Does added CBAM improve a strong backbone with the same head and recipe? | One or two best Phase-3 models, same seed/settings; retain negative outcomes. Repeat the strongest pair at seed 43 if budget allows. |
| Phase 5 | Can one diagnosed limitation improve balanced performance? | At most two predeclared changes; augmentation or balancing motivated by errors, one factor at a time. |
| Phase 6 | Do complementary models improve probability fusion? | Equal two-member average first; at most one fixed 3-member extension. No hundreds-of-weights search. |
| Phases 7–9 | What does the frozen final model achieve and where does it focus? | Freeze all choices before one test evaluation; representative Grad-CAM with correct/incorrect cases. |

## Common settings

| Setting | Fixed decision |
|---|---|
| Data | Existing strict lesion-disjoint manifest/hash; seven classes; natural training references. |
| Inputs | PIL RGB; resize directly to 224×224, bilinear interpolation with antialias; ImageNet mean `[.485,.456,.406]`, std `[.229,.224,.225]`. Aspect distortion is a disclosed common limitation. |
| Augmentation | Training horizontal flip p=.5 only; deterministic validation. Stronger augmentation is a later controlled factor, given prior regression. No crop/hair-removal/CLAHE/TTA in the initial comparison. |
| Pretraining | Exact torchvision ImageNet enum in each config; record downloaded weight hash. V1/V2 training recipes differ across backbones; disclose rather than claiming identical pretraining. |
| Head | Feature map → optional identity → GAP → native ConvNeXt normalization where needed → dropout .2 → linear seven logits. No pretrained classifier. All layers fine-tuned from epoch 1. |
| Added attention | None in Phase 3; native architecture SE/stochastic-depth remains. Later CBAM: final feature map, reduction16, spatial kernel7, same head/recipe. “None” means no **added CBAM**, not absence of all internal attention. |
| Imbalance/loss | `w_c=sqrt(7009/(7*n_c))`, normalize weights to arithmetic mean1; counts from full training partition only. Weighted cross entropy; no oversampling/focal/smoothing. |
| Optimizer | AdamW; backbone3e-5, newly initialized head/CBAM1e-4; weight decay1e-4, betas(.9,.999), eps1e-8. Native ConvNeXt norm belongs to backbone LR group. No silent optimizer overrides. |
| Batch | Training microbatch16, accumulate2 → effective32; shuffle with logged epoch seed/order; drop_last=True removes one shuffled training reference per epoch (7008 used). Validation batch16, no shuffle, no dropping. |
| Loss accumulation | Optimize sum of weighted per-image CE divided by sum of target-class weights over the **whole effective batch**; do not average independently normalized microbatch losses. Validation loss uses the same global weighted numerator/denominator. Record exact implementations. |
| Budget | Maximum20 epochs, early-stop after5 consecutive epochs without a strict macro-F1 increase; one initial seed42. No underperforming-model early cull before this common stop rule. |
| Scheduler | ReduceLROnPlateau max macro-F1, factor.5, patience2, threshold0, threshold_mode=abs, cooldown0, min_lr1e-7; step once after validation. |
| Selection | Earliest epoch with maximum full-validation macro-F1; accuracy/class metrics do not choose another checkpoint. Overall rank macro-F1 → accuracy → lower measured inference cost for exact ties. |
| Reproducibility | Seed Python/NumPy/Torch/workers; deterministic algorithms, cuDNN benchmark=False; CUDA deterministic workspace configured before CUDA initialization. AMP fp16 with GradScaler; record CUDA/runtime/device and optimizer updates. Resume stores all RNG/optimizer/scaler/scheduler state. |

## Feasibility and amendments

Before runs, propose a brief GPU memory/timing preflight with question, command, monitoring and output paths. It is not executed in Phase 2. First verify CPU runner, loss accumulation, partition safety and resume logic. If any selected model cannot fit microbatch16, propose a **common microbatch8/accumulation4 amendment for all not-yet-started comparisons**; otherwise rerun affected controls or report a separate recipe. Accumulation does not equalize BatchNorm microbatch statistics. Never silently lower one model's batch or input size.

Publish config hashes and a change note before changing recipe v1. Record actual timings/memory; do not infer GPU fit from parameter count. No ImageNet downloads or new HAM10000 result is needed to fix this plan.

## Interpretation and paper evidence

Save complete packages even for failed/negative experiments. Comparison table includes protocol, added/native attention, exact pretraining, seed, epochs/best epoch, accuracy, macro P/R/F1, per-class scores/support, checkpoint hash, parameters, training time and measured inference cost. Show raw/row-normalized matrices, available train/validation curves, class charts and common-scale comparison plots. New rows stay separate from historical best-of-grid records.

Treat +.005 macro-F1 as a practical signal for an attention follow-up, **not** statistical significance. Always report accuracy, melanoma recall and minority-class changes; do not conceal trade-offs. Repeat the leading matched pair at a second seed when feasible. No ensemble is final solely because it raises majority-driven accuracy. Grad-CAM indicates score influence, not clinical validation; images, labels, confidence, target layers and selection criteria must accompany overlays.
