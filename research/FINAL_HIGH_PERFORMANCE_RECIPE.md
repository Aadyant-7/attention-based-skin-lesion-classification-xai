# FINAL HIGH-PERFORMANCE RECIPE (prepared, NOT launched)

Frozen architecture: [{"run_id": "s03_efficientnet_b0_none_exploratory_seed42", "model": "efficientnet_b0", "weights": "EfficientNet_B0_Weights.IMAGENET1K_V1", "checkpoint": "checkpoints/structured/s03_efficientnet_b0_none_exploratory_seed42/best.pt", "checkpoint_sha256": "6b7a270de98912a7f7a70ca3c1a78e55f8b5a0fe1406762760699cb23d4ca39c"}, {"run_id": "s06_convnext_tiny_none_exploratory_seed42", "model": "convnext_tiny", "weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1", "checkpoint": "checkpoints/structured/s06_convnext_tiny_none_exploratory_seed42/best.pt", "checkpoint_sha256": "92381249b6486f54a8db6c7d808e5d421bf7ba1c0f8c7ea370d24542ec247ef7"}, {"run_id": "s10_efficientnet_v2_s_none_exploratory_seed42", "model": "efficientnet_v2_s", "weights": "EfficientNet_V2_S_Weights.IMAGENET1K_V1", "checkpoint": "checkpoints/structured/s10_efficientnet_v2_s_none_exploratory_seed42/best.pt", "checkpoint_sha256": "120b621b6ddc280c0929273121b149e4c7e90aec4c088c4da0c5b41a1dce62f5"}]
Frozen member weights: [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]

| Technique | Decision | Reason |
| --- | --- | --- |
| ImageNet transfer learning | ADOPT | Strong local and literature evidence; fresh external initialization for strict confirmation. |
| Full fine-tuning / progressive unfreezing | ADOPT / OPTIONAL | Full fine-tune retained; progressive unfreezing optional, not silently combined. |
| Discriminative LR | ADOPT | Preserve3e-5 backbone/1e-4 head philosophy. |
| Class-weighted cross-entropy | ADOPT | Train-only sqrt inverse frequency; no validation fitted class weights. |
| Focal/class-weighted focal | REJECT for default | Literature support but no matched local benefit; historical packages confound effects. |
| Minority/class-specific augmentation | OPTIONAL | Potential major gap; controlled mild spatial augmentation only after explicit approval, not paper copied strengths. |
| Mixup | OPTIONAL | Literature-supported regularization, no demonstrated local effect; not in frozen default. |
| CutMix | OPTIONAL | Unverified local effect; not combined with Mixup by default. |
| Dropout | ADOPT | Common head .2 unchanged. |
| Weight decay | ADOPT | AdamW1e-4 retained. |
| Label smoothing | OPTIONAL | Could conflict with class weights; not default. |
| Warmup/cosine/OneCycle | OPTIONAL | Different optimization package; not stack with plateau scheduler. |
| ReduceLROnPlateau | ADOPT | Accuracy mode; same factor .5, patience2; longer early-stop allowance. |
| Gradient clipping | ADOPT | Unscaled global norm1 for numerical protection in future strict recipe, not changed in screening. |
| EMA | OPTIONAL | Extra state/selection complexity; no current evidence. |
| Increased resolution | REJECT for default | Current224 matched; TTA failed and S14deferred, no demonstrated gain. |
| Weighted soft voting | ADOPT if selected | Exact frozen weights only, otherwise equal reference. |
| CBAM/attention | REJECT as forced final add-on | PreserveS05 mixed/near-neutral ablation; native attention retained. |
| Grad-CAM/XAI | ADOPT later | Correct/incorrect development and final test examples after final freeze. |

## Coherent default
Fresh ImageNet weights; strict lesion-disjoint train/val;224px/ImageNet normalization; horizontal flip.5; fullfine-tune; effectivebatch32; AMPtraining with finite guards and FP32validation from start; weightedCE; AdamW discriminativeLR3e-5/1e-4; dropout.2; decay1e-4; gradientclip1; plateau scheduler; cap50/minimumreview15/patience12. Record this new recipe version separately from screening. Optional techniques excluded from default, require a deliberate amendment. Ensemblemember weights frozen; no strict-validation weight search. Checkpointselection uses frozen ensembleaccuracy for final ensemble, with independentmacroF1 and member states saved.

No guide/paper result claimed for this unexecuted recipe. Strict adaptation can change performance. No next backbone.
