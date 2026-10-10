# Separate paper-guided image-level study — v1

Approved scope on 10 October 2026: prepare Phase 1 only. This is a new study from fresh external pretrained weights, not a continuation of S83/S97 or a change to their reported results. No GPU run is authorized by this plan or the protocol preparation command.

## Purpose and reporting

Investigate paper-reported transfer-learning, training-only balancing and controlled ensembles under the guide's approximately **70/15/15** proportions. Aim for higher image-level accuracy; no 93%, 95% or 97% result is promised. Prior project results are known, so label this **post-development image-level evaluation with lesion overlap**, not a pristine independent or lesion-independent test.

Preserve the previous strict studies and the frozen S83 result: **94.011976% exploratory validation / 87.558217% original-cohort test audit**. Tables must identify their protocol. New results never replace historical scores.

## Phase 1 — One frozen original-image split

- Use all 10,015 original HAM10000 image identities. Sort image IDs canonically before splitting.
- Stratify by the seven diagnoses with seed 42, using a 3,006-image holdout then dividing it into 1,503 validation and 1,503 test originals.
- Counts: 7,009 training, 1,503 validation, 1,503 test. These match the previous study's counts and the guide's proportions; image assignments are new.
- Permit different original photographs of one lesion in different partitions. Measure natural lesion overlap; do not select a seed or deliberately allocate shared lesions to improve accuracy.
- Keep original image IDs unique and disjoint. Split before creating training-only augmented copies or oversampled exposures. Validation/test retain natural class proportions.
- Freeze manifest hash, metadata hash, class order, algorithm and dependency versions before training. Refuse silent resplitting or overwriting partial preparation.
- Read metadata labels for stratification/counts and filenames for availability only. Do not decode images, infer probabilities, train models or calculate model accuracy in this phase.

This differs from the old exploratory manifest, which reshuffled only the old development pool and retained the lesion-disjoint test. The new manifest also reassigns original-test images. Old assignment files remain unchanged.

## Phase 2 — Prepare one B3 adaptation

Primary reference: [DermAI 1.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10573070/), sections 3.4–3.6. Reported components include ImageNet pretraining, two 4,096-unit dense layers, categorical cross-entropy, learning rate 1e-4, batch 16, dropout 0.5 and a 50-epoch allowance. Its K10/augmented-data evaluation differs from our required 70/15/15 protocol.

Verify the paper's input preprocessing, augmentation, balancing, optimizer and unfreezing details. Record missing information as explicit implementation assumptions. Framework/weight differences must be disclosed. Training-only balanced sampling is a proposed adaptation if the paper's balancing cannot be reconstructed, not a verified exact recipe. Do not combine focal loss, Mixup and CBAM into the initial plain-B3 baseline.

S16 plain B3 failed during epoch 12 after 11 valid epochs. S32 B3+CBAM completed 43 strict-development epochs at 86.02794%; its recipe was substantially different. Neither establishes an exact replication of DermAI's reported 97.01%.

Before launch, prepare a separate runner/config with the new protocol hash. Existing `research.train` and `research.strict_protocol` enforce the old test cohort and cannot be silently repurposed. Use fresh external pretrained weights; **never initialize from old project checkpoints or reuse old fitted models, embeddings or probabilities**, because old training identities enter this new test.

## Phase 3 — Bounded training and comparator

Provisional budget, to be fixed in the actual proposal/config: inspect trajectory around epoch 15, minimum 20 epochs unless numerical failure, 30 planned epochs, maximum 50 only under a continuation gate declared before launch. Use meaningful-improvement stopping and LR scheduling; keep numerical accuracy-best/F1-best/latest states even when tiny gains do not reset patience. CPU preflight verifies correctness/feasibility, not expected accuracy.

Train fresh ConvNeXt-Tiny on the identical new manifest as the proven-family comparator. Practical recipe comparisons must disclose differences; they are not pure architecture ablations when heads/sampling differ.

## Phase 4 — Controlled ensemble and attention

Analyze validation error complementarity. Equal B3 + ConvNeXt probability fusion first; at most two predefined weighted alternatives. A third model requires evidence that its expected contribution justifies compute. No automatic architecture queue.

Add CBAM through a matched attention experiment on the stronger backbone. Describe its presence separately from evidence that it improved performance. Grad-CAM explains the actual final method after selection.

## Phase 5 — Freeze and one new-protocol test session

Use validation only for model, checkpoint, budget, ensemble weights and preprocessing decisions. Freeze them before one final evaluation of this new test partition. No rerun or new method after seeing its score. Report full-cohort accuracy, macro precision/recall/F1, weighted-F1, per-class support/scores, melanoma/akiec recall and raw/normalized confusion matrices. Report seen/unseen-training-lesion subgroup scores descriptively from the same saved predictions; never select on those test subgroups.

Save model probabilities, predictions, best/F1/latest checkpoints, logs, histories, curves, comparison figures and registry entries. New serious experiments belong to this study's registry and the master registry under the distinct protocol ID; no metric rows exist before training.

## Launch policy

Every GPU run needs a concrete proposal, command, expected cost, manual monitoring command and artifact paths, followed by user approval. After launch and logging/checkpoint confirmation, stop interaction immediately. Phase 1 does not alter that policy.
