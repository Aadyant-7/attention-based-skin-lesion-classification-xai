# Paired targeted ConvNeXt intervention

Motivation: strict-development error audit found 35 melanoma and18 BKL images missed by all three constituents, plus30/14 respectively recoverable by a constituent. Blind fusion changes and adding the aggressive models did not help. This tests a training exposure hypothesis, not a new architecture.

## Immutable inputs

- Parent: S29 ConvNeXt-Tiny standalone accuracy checkpoint, epoch33, SHA256 `ffe3a7f00c5371824fa46a8f9e5f864f53e15fcc533f5bf0263648f8573a1e93`.
- Existing7009 training/1503 validation images, lesion-disjoint. Revealed original1503 test excluded; no test inference.
- Preserve224/ImageNet normalization, native LayerNorm/simple dropout.2 head, horizontal flip, sqrt-weighted CE, AdamW decay.0001, effectivebatch32/micro16 and AMP/FP32 validation.
- Both arms use fresh optimizer and gentle backbone/head LRs3e-6/1e-5. This is a new warm-start intervention, not a continuation of the original optimizer or strict frozen method.

## Paired intervention and compute cap

S37 control:5 new epochs. Add1548 uniformly drawn extra training indices to the full training list, then shuffle. S38 target: replace those extra indices with one additional copy of each MEL/BKL training image. Both arms have the same8544 examples/267 optimizer steps per epoch after terminal-batch truncation. Their only planned difference is class composition of the extra exposure; effective sampling frequencies change explicitly. Both begin from the exact same parent weights and seed. This is not a pure estimate of repeated sampling versus no repetition; it isolates targeted versus uniform extra exposure.

Target pilot5 epochs, optional extension to10/15, NEVER above15. Together the maximum is20 new training epochs. Runtime estimate20–30minutes if all allowed epochs are used; actual device conditions may differ. Failed pilot stops after5, so total10. No third arm, focal/Mixup package, new backbone or weight search.

## Gates before seeing results

Compare fixed B0 + arm accuracy-winner + V2-S equal ensembles against both the original frozen reference and matched control:

- At5 target epochs: accuracy at least +.005 (0.5pp), macro-F1 not lower, mean MEL/BKL recall at least +.02, neither focus class recall declining by >.01 and NV recall declining by >.01.
- At10: retain initial gate and require at least another+.002 ensemble accuracy over pilot5. Otherwise stop.
- Numerical best accuracy/F1 checkpoints update independently. Parent is retained as epoch0 if no new accuracy improvement. No ensemble-epoch search; each arm uses its standalone raw-accuracy winner.
- Hard caps are independent of patience or scheduler; tiny gains cannot extend training. LR scheduling has a meaningful.002 threshold.

## Outputs and recovery

Start/recover: `.\.venv\Scripts\python.exe -u -m research.short_screening.targeted_finetune --batch` from project root.

Results/logs/state: `results/short_screening/targeted_finetune_v1/`; checkpoints: `checkpoints/short_screening/targeted_finetune_v1/`. Each arm saves best/latest/macro-F1 states, full RNG/optimizer/scheduler/scaler recovery, history/LRs, metrics, prediction/probability files, per-class scores, confusion matrices, curves and fixed-ensemble comparisons. The master skips completed work, reattaches live workers by PID/create-time, resumes saved epoch boundaries, closes out both arms, publishes and stops. Original source artifacts are never overwritten.

Report as post-test strict development; historical model selection used the validation cohort. No result is a fresh independent test estimate. No promise of92% test accuracy.
