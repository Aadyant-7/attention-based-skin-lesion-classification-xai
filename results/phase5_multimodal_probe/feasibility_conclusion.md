# Phase 5A validation-only feasibility conclusion

The fixed lesion-level split and cached 1,280-dimensional post-CBAM image embeddings were reused. Only `age`, `sex`, and `localization` were joined by `image_id`; the ID was not a predictor. The age imputer/scaler and categorical encoders were fit on the 7,009 training rows, then applied to the 1,503 validation rows. No test metadata, features, or predictions were used.

Metadata alone was weak (accuracy 0.278776; macro F1 0.185897). In single-field Logistic Regression ablations, **age** contributed most: adding it to image embeddings changed validation accuracy from 0.838323 to 0.840985 (+0.002661) and macro F1 from 0.734722 to 0.738495 (+0.003773). Localization gave a smaller macro-F1 increase (+0.001817); sex lowered macro F1 (-0.001798). Adding all three gave macro F1 0.732722, below image embeddings alone.

The best full multimodal classifier was the small MLP (accuracy 0.842315; macro F1 0.745492). The best metadata-aware fusion used 90% weighted CNN flip-TTA probabilities and 10% MLP probabilities (accuracy 0.845642; macro F1 0.770951). The highest metadata-aware accuracy, 0.846307, used an 80%/20% mix and had macro F1 0.763740.

The existing image-only validation leaders remain ahead: accuracy 0.857618 and macro F1 0.771810. The best metadata-aware accuracy is **0.011311 lower**, and its best macro F1 is **0.000859 lower**. Neither reaches the requested +0.015 improvement threshold. **Stop after this feasibility probe; a full multimodal CNN training experiment is not justified by these validation results.**

Against the best image-only macro-F1 ensemble, the best metadata-aware fusion reduces mel↔nv errors from 104 to 101 and bkl↔mel from 32 to 31. bkl↔nv stays at 34 and akiec↔bcc stays at 12. Mel per-class F1 nevertheless falls slightly (0.5465 → 0.5449); nv rises (0.9213 → 0.9232), while bcc and bkl also fall slightly. The changes are too small and mixed to establish a generalization benefit.
