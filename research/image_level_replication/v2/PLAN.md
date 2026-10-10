# Active phased plan V2 — K10-inspired adaptation

10 October 2026. User permits changing split proportions to follow relevant papers more closely. This new V2 plan supersedes the earlier fixed-70/15/15 plan; it does not edit V1 or historical results. Only revised Phase 1 is currently authorized.

## Phase 1: freeze all partitions and record the literature decision

Canonical original-image order, seven-class stratified ten-fold split, shuffle seed 42. Each original is an outer assessment case exactly once. Within each fold, stratify the remaining 90% into 90% inner fitting / 10% inner validation using seed 42 plus fold ID. Effective sizes are 8,111–8,112 / 902 / 1,001–1,002, approximately **81/9/10**.

Allow and measure naturally occurring lesion overlap. Keep original image IDs disjoint within each fold, split original images before augmentation/balancing, and leave validation/assessment at natural class proportions. Preserve all strict, exploratory, S83 and V1 artifacts.

Record source/manifest hashes, label order, versions, per-fold class counts and overlap, historical preservation and reproducibility. Metadata labels serve allocation/counting only; no image opening, GPU use, training, inference, predictions, score search or test evaluation in Phase 1.

Protocol choice is based on the paper's stated K10 structure. CPU metadata cannot establish which split will yield the highest accuracy. Missing/conflicting reference details are explicitly listed in `PAPER_PROTOCOL_REVIEW.md`.

## Phase 2: prepare one plain B3 adaptation

Use DermAI as the single primary recipe reference. Verify head, preprocessing/input resolution, weight version, augmentation, balancing, optimizer and fine-tuning details; separate verified fields from assumptions. Keep its reported cross-entropy/head/dropout/training settings where reproducible. Do not silently stack the failed aggressive focal/Mixup/CBAM package onto this first baseline.

Use fresh external pretrained weights. Old project fine-tuned checkpoints, fitted metadata models, cached features or probabilities cannot enter this new evaluation: their prior training images occur in outer assessment folds. Every model/fold needs its own initialization and artifact namespace.

Prepare a **fold-aware isolated runner**, which accesses only inner train/validation for training and selection, checks the specific fold hash/class order, and saves standard research artifacts. Existing strict or V1 runners cannot be silently repurposed. The first proposal uses fixed **fold 0** for development and never scores its outer assessment during discovery.

## Phase 3: bounded development and complete evaluation decision

The epoch policy still needs to be fixed in the actual config/proposal. Current working budget: evaluate trajectory at around 15, allow minimum 20 unless numerical failure, plan 30, cap 50 under a predeclared continuation gate. Use meaningful-improvement counters/LR scheduling and raw accuracy-best/F1-best/latest checkpoints. This remains an adaptation of the paper's stated 50-epoch training, not an exact copy.

Do not automatically launch ten folds. First obtain a useful, finite inner-fold B3 result, then a fresh ConvNeXt comparator if justified. Select components and recipes on development evidence, not outer scores. Freeze candidate set, preprocessing, epoch/checkpoint rules and fusion rule before complete outer assessment.

A model fitted on inner training uses 81%, not 90%. If matching the paper's larger fitting fraction is worth the additional cost, define a fixed epoch budget from inner validation and start a **fresh outer-90% refit**, including that fold's inner-validation originals and excluding its outer assessment. This refit has no outer-driven early stopping/checkpoint selection. It is another explicitly approved run, not automatic continuation. Report the actual fit fraction.

Global recipes developed on these data are post-development choices; internal CV is not an external untouched estimate. Apply the same predeclared decision rules within folds and disclose development history.

## Phase 4: bounded ensembles and attention

Equal B3 + ConvNeXt fusion first. At most two predefined weighted alternatives on inner validation if justified by complementarity. A third model needs evidence and its own GPU proposal. Train/compare CBAM as a named adaptation with a matched control if claiming an accuracy gain. The paper's reported attention cannot be equated with CBAM without implementation evidence.

Ensemble evaluation must use constituent models trained for the **same outer fold**. For an image assigned to fold k, use only fold-k constituent probabilities. Combining other-fold models contaminates OOF evaluation because those models may have trained on that image.

## Phase 5: complete reporting

For a K10 claim, obtain all ten outer assessments with one frozen-method session per fold and exactly one OOF prediction per original. Report pooled OOF accuracy/macro precision/recall/F1/weighted-F1, class support/recall/F1 and raw/normalized confusion matrices. Also report all fold scores with mean/std. Do not select the best fold or mix checkpoint epochs to create a composite score.

If only one or a few folds are completed, report those as **partial evaluation**, not ten-fold performance. Save full predictions/probabilities, models, histories, plots, logs, registry entries and costs. Descriptive seen/unseen-lesion subgroup metrics can accompany the same OOF predictions without guiding further outer-score tuning.

Grad-CAM/XAI and final writing use the actual frozen method. Preserve the prior strict and S83 evidence, and distinguish any new post-development image-level result. No guaranteed 93%, 95% or 97% accuracy.

## GPU authorization

No GPU training is authorized by Phase 1 or by creating all fold manifests. Before any run: propose question, recipe, fold, budget/runtime/VRAM, exact start/monitor commands and checkpoint/log/result paths; wait for approval. After logging/checkpointing are confirmed, stop immediately. No epoch polling or automatic model/fold queue.
