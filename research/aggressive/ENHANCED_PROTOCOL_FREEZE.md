# Enhanced protocol freeze — post-test development study

Option B selected BEFORE any enhanced outcome: reuse existing strict lesion-disjoint development split,7009train/1503validation,seed42. The original1503test images/labels are excluded from all enhanced training, selection and XAI. Full HAM10000 is NOT used. All lesion overlaps zero; immutable SHA db1ce9f8b83f176001dd89fd332fb1668c7d2ba5f58d77157d293a09e2c2e696. No new split or subset cherry-picking. Development validation has been used previously; it is NOT a new independent test. The original rigorous held-out result86.7598% / macroF1 .794473 is permanently preserved separately.

Cost rationale: three models × up to50epochs instead of nine CV jobs. Prior local epoch costs roughly45–115s imply2–5hours for one queue versus6–15hours for3folds, before richer augmentation/FP32 BN/XAI overhead. Disposable preflight will refine an estimate, not select models or protocols. One fold cannot reproduce the paper's3fold mean/SD. Sequential execution selected due approximately4GB available host RAM and one8GBGPU.

Frozen architecture: B3+existing channel/spatial CBAM after final convolution, DenseNet201,ResNet101; fresh explicit ImageNet weight versions in config.json. Primary voting .40/.40/.20, equal1/3 secondary baseline only. No weight/member search. Finish the three branches and report the real result; stop whether above or below93%.

One combined recipe: FP32 stable log-softmax weighted focal gamma2.2; training-only bounded inverse-frequency weights (no manually tuned source multipliers); Mixup alpha.2/p.3 with a convex blend of correct class-specific focal objectives. Natural class counts retained; targeted spatial/color/erasing augmentation only in training; strongerAKIEC/DF/VASC and moderateBCC/MEL. Hue capped.05 and erasing1–5% to limit morphology damage. Norm mean [0.7630306432877934, 0.5459358876080843, 0.5705884149341853],std [0.1402932921658137, 0.15144422149154305, 0.16872958982765698] computed on ALL training pixels only, not copied source statistics.

Custom GAP/d→512→256→7 head,BN/ReLU,dropout.65/.55/.45 across layers (NOT across epochs). AdamW/model LRs1e-4/7.5e-5/5e-5; decay.0015; clip.5; StepLR.7/10. Head/attention only epochs1–2,last backbone stage3–5,full fine-tune6+. Frozen backbone BN stats remain frozen; active BN isFP32. Max50; patience12 begins at25,delta.0003 for stopping reference; always preserve true earliest maximum accuracy even if smaller gains don't reset patience. Independent bestF1/latest retained. Effectivebatch32; terminal33 preserves ALL images.

Numerical safety: oldB3 overflow atfeatures.7.0.block.0.0; oldFP32 probe was not authorized (false flag means not performed, not failure). BF16 uses FP32-range exponents; loss/BN/validationFP32. BF16 requires GPU support and finite forward/backward preflight. Predeclared one fullFP32 recovery from latest finite boundary; micro<=16 if needed, sameeffectivebatch. Every failure and precision amendment retained; no invalid sample skip or NaN/Inf clamp. Disposable fit-probe weights are discarded; actual runs start fresh.

Accuracy winner rule is independent per branch; source trained simultaneously/validated epoch ensemble, while our sequential queue cannot reproduce that selection trajectory. This is an adapted pipeline, not exact replication. Optional sourceLIME is omitted by default; complete Grad-CAM, actualCBAM maps and bounded validation-only occlusion examples are included. These visualizations are descriptive and do not prove clinical reasoning.

Original results and registry rows are hash-protected. New files only under research/aggressive,results/aggressive_enhanced,checkpoints/aggressive_enhanced. Post-test enhanced scores are labeled explicitly in every table/report. No further performance round or test evaluation is scheduled.

Pre-launch resource amendment: microbatch32,effective32; defaultprecisionbf16; model-specific FP32 initial overrides if present inconfig. Disposable fit-probe conservative maximum estimate 11.3hours plus I/O/validation/XAI; no outcome seen.

Pre-launch resource amendment: microbatch16,effective32; defaultprecisionbf16; model-specific FP32 initial overrides if present inconfig. Disposable fit-probe conservative maximum estimate 11.8hours plus I/O/validation/XAI; no outcome seen.

Pre-launch resource amendment: microbatch16,effective32; defaultprecisionbf16; model-specific FP32 initial overrides if present inconfig. Disposable fit-probe conservative maximum estimate 4.1hours plus I/O/validation/XAI; no outcome seen.
# Live stopping amendment

The user authorized meaningful-improvement stopping on 2026-10-04. `stopping.py` and `MEANINGFUL_STOPPING_AMENDMENT.md` now supersede the original patience/min_delta stopping fields below: minimum25/maximum50; meaningful accuracy +.002 OR macro-F1 +.003; patience12; late plateau10 after30 with LR opportunity and three-epoch settling. Raw best checkpoint ranking remains unchanged. The original config is retained byte-for-byte for recovery identity; all training/data/architecture fields remain unchanged.
