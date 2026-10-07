# S70-S72: multi-image information and nonlinear-head closeout

Completed 6 October 2026; verified and documented 7 October 2026 after the usage interruption. No GPU run, image inference, original locked-test scoring, checkpoint change or parameter search.

## S70/S71: known same-lesion companion images

One fixed arithmetic mean pools the existing ensemble probabilities of distinct images with the same lesion ID **inside validation only**. No training/test companions, diagnosis labels, confidence weights or selected image subsets enter prediction. Singleton lesions retain their original scores. This changes the input contract: the method requires multiple images and known grouping. It is not a single-image benchmark improvement or a replacement for the frozen test method.

| Same-cohort endpoint | Original accuracy / macro-F1 | Multi-image accuracy / macro-F1 | Gained / lost | Decision |
|---|---:|---:|---:|---|
| S70 exploratory S53, 1,503 images | 93.6128% / 0.886857 | 93.7458% / 0.888382 | 5 / 3 | Material gate failed; melanoma recall declined |
| S71 strict S31 validation, 1,503 images | 90.1530% / 0.843486 | 91.0845% / 0.854216 | 29 / 15 | Conditional strict multi-image gate passed; retain as separate research evidence |

Exploratory validation contains 1,429 lesions: only 70 have multiple available validation views, comprising 144 images, at most three views. Strict validation contains 1,122 lesions: 292 multi-image lesions, comprising 673 images, at most five views. More actual companion views were available in the strict cohort. This does not prove view availability is the sole cause of the different gains.

S70 melanoma recall falls 82.0359% -> 81.4371% (one fewer correct melanoma). BKL gains one and nv gains two. S71 melanoma recall rises 60.8434% -> 65.0602% (seven additional melanoma images), akiec 74% -> 76%, and BCC 88.3117% -> 94.8052%; df loses one correct image.

The fixed advancement rule required at least eight net correct images, +0.005 absolute accuracy, nondecreasing macro-F1 and melanoma recall in **both** protocols. It did not pass. Do not search mean/max/product/confidence-weighted variants to rescue this result. S53 remains the retained single-image exploratory candidate; S31's original strict/test reports remain unchanged.

### Lesion-weighted endpoint is a different denominator

We also score each lesion once. The single-view comparator is the lexicographically first image ID, fixed without labels/confidence; the candidate uses every available validation view in that lesion. These scores are not the 1,503-image benchmark:

| Cohort | Lesions | One fixed view accuracy / macro-F1 | Multi-image accuracy / macro-F1 |
|---|---:|---:|---:|
| Exploratory | 1,429 | 93.9118% / 0.885195 | 93.9118% / 0.885849 |
| Strict validation | 1,122 | 92.3351% / 0.854725 | 92.8699% / 0.859498 |

Paired lesion-cluster bootstrap, 10,000 replicates/seed42, gives image-weighted accuracy-gain intervals of [-0.2012, +0.5263] percentage points for S70 and [+0.0654, +1.8256] for S71. These are descriptive intervals after repeated development selection, not independent confirmation or a general superiority claim. Equal pooling assumes the images concern the same lesion; it cannot be applied across arbitrary patients or images sharing a predicted class.

Literature motivation: [Yap et al., multimodal skin lesion classification](https://doi.org/10.1111/exd.13777) combined clinical/dermoscopic modalities and metadata. Our repeated-dermoscopy arithmetic mean is a different method, not replication of its architecture, cohort or performance.

## S72: train-only nonlinear frozen-feature classifier

One fixed comparison reused ImageNet-1k ConvNeXt-Tiny square224 feature caches. The control is fold-local StandardScaler + weighted logistic regression C1. The candidate adds an RBF Nystroem map (gamma1/768, 512 components, seed42) and fold-local scaling, with the same logistic head and train-derived weights. The three lesion-group folds cover all 7,009 development-training images exactly once; no evaluated lesion is present in that fold's training data.

| Train-only group-OOF classifier | Accuracy | Macro-F1 | Melanoma recall |
|---|---:|---:|---:|
| Linear control | 75.8025% | 0.573473 | 50.2571% |
| Fixed nonlinear candidate | 75.8025% | 0.557074 | 48.0720% |

Candidate fold net changes: +11, +13, -24. It gains and loses 582 predictions each: no pooled accuracy improvement, lower macro-F1 and melanoma recall. The predeclared +1 percentage point/nondecreasing class-score gate failed. **No development-validation feature cache was loaded or scored; no GPU proposal follows from S72.** This rejects the fixed approximate-kernel recipe, not every nonlinear model. See [scikit-learn's Nystroem documentation](https://scikit-learn.org/stable/modules/kernel_approximation.html#nystroem-method-for-kernel-approximation).

## Saved evidence and verification

- `results/short_screening/s70_s71_same_lesion_multiimage/`: fixed plan, logs, per-protocol image/lesion endpoints, complete prediction/score CSVs, class and gain/loss tables, confidence intervals, source hashes, raw/normalized confusion matrices and PNG/PDF comparisons.
- `results/short_screening/s72_nonlinear_frozen_feature_head/`: fixed plan, train-fold assignments/results, complete OOF predictions/probabilities, metrics/class/confusion/comparison figures, source hashes and verification. Ignored fitted heads live in `.cache/s72_nonlinear_frozen_feature_head/`.
- The master registry adds separate multi-image and train-only feature-head record kinds/protocols. They must not be ranked as comparable standalone trained CNN results.

Checks passed: ID/class alignment, original-test identity/group exclusion, singleton preservation, label-free pooling, grouped fold separation, finite normalized scores, source-file integrity, and independent recomputation of saved metrics/confusion matrices. Existing checkpoints/predictions were preserved. Processing runtimes were 12.6 seconds for S70/S71 and 14.1 seconds for S72, excluding interpreter startup.

Original held-out test accuracy remains **86.7598%**. Highest observed original single-image exploratory candidate remains S63 **93.6793%**, while the retained provisional candidate remains S53 **93.6128%**. S70's **93.7458%** belongs to a different multi-image input protocol and was not accepted.
