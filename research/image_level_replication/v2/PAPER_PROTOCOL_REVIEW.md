# V2 paper-protocol review

10 October 2026. Phase 1 preparation only; no training, inference or model accuracy is produced here. V1 and all historical results remain preserved.

## Primary reference: verified and unresolved details

Source: [DermAI 1.0, Sanga et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10573070/).

| Reported detail | Location / replication consequence |
|---|---|
| HAM10000; 112 x 112 inputs; augmentation for balancing, then class weights | Sections 3.2-3.3; exact augmentation magnitudes and multiplicities are unresolved. |
| ImageNet-pretrained backbones, including EfficientNet-B3; two 4,096-unit ReLU dense layers and seven-class softmax | Section 3.4.1; precise pooling, normalization and unfreezing need verification. |
| Learning rate 0.0001, categorical cross-entropy, batch 16, dropout 0.5, 50 epochs, TensorFlow 2.0 | Section 3.5; optimizer and checkpoint-selection details are not established here. |
| 80/20 partition versus default K10; B3 accuracy 97.01% | Sections 3.3, 3.6 and 4.5 conflict; not a single fully specified split recipe. |

Original-image augmentation lineage, lesion grouping, exact fold assignments and usable reproduction code remain unverified. The narrative places augmentation before partitioning; it does not establish whether related copies crossed evaluation boundaries. We cannot conclude leakage occurred or certify its absence. These ambiguities prevent an exact replication or a promise of 97% performance.

## V2 decision: K10 with separate selection data

Freeze one `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` assignment on canonically sorted original image IDs. Stratify by diagnosis, allow natural lesion overlap, measure it, and avoid seed or overlap searches.

Within each outer 90% development pool, reserve a stratified 10% inner-validation subset. Actual fitting therefore uses approximately **81% training / 9% inner validation / 10% outer assessment**:

- Five folds: 8,111 training / 902 inner validation / 1,002 assessment originals.
- Five folds: 8,112 training / 902 inner validation / 1,001 assessment originals.

This is our explicit adaptation, not the paper's claimed 90% fitting fraction. Inner validation supports checkpoint selection, stopping and bounded ensemble decisions without using the outer assessment. Original images are disjoint within every fold. Augmentation and balancing operate only on training originals, after partitioning. Evaluation originals retain natural class counts.

An optional fresh refit on the entire outer 90% can use a fixed epoch budget chosen solely from inner validation. That adds compute and requires a concrete proposal and GPU approval; it is not authorized by Phase 1. It must not be mixed silently with the 81% fitting results.

## Assessment and ensemble rules

Use fresh external pretrained initialization for every fold and every component. Resume only the same new run with its frozen configuration. Previous project checkpoints, fitted embeddings and probabilities are ineligible because their training images enter the new assessment partitions.

Each original must receive exactly one out-of-fold prediction from a model that excluded its outer fold. Ensemble components must use the **same outer fold**. Do not average all ten fold checkpoints for each assessment image: most of those models trained on that image.

Report pooled out-of-fold metrics, all ten fold scores, and fold mean/standard deviation only after completing all ten assessments. A single pilot or best fold is not a K10 result. Outer scores must not select checkpoints, weights, configurations or further methods; freeze the development workflow before their disclosure. Use the same fold assignments and prespecified comparison rules across candidate methods.

Full K10 means about ten independent training runs per backbone, plus any approved refits. Phase 1 freezes metadata only. Phase 2 must first propose one bounded inner-fold pilot and its GPU cost, without outer assessment, rather than silently launching a ten-run queue.

## Attention implementation caution for Phase 2

Our mathematical inspection of the paper's equations 3-4 gives a caveat: if each model's seven probabilities sum to one, averaging them across classes yields `1/7`. Multiplying every class score by that constant cannot change its argmax. The written formula alone therefore does not establish a useful attention mechanism. A different implemented layer may exist, but that implementation is unverified. CBAM would be a separately identified adaptation, not an exact reproduction of this formula.

## Interpretation and preservation

Label future results **post-development internal image-level cross-validation with lesion overlap**. Earlier test outcomes are already known. This protocol cannot establish pristine external-test or unseen-lesion generalization, and it does not replace the previous 87.558217% S83 test audit or 94.011976% exploratory validation result.

The change increases the training pool and matches the reference's K10 structure more closely. Neither metadata preparation nor CPU preflight demonstrates that accuracy will increase. Preserve V1's 70/15/15 manifests, policy, artifacts and results; V2 remains a separate namespace.
