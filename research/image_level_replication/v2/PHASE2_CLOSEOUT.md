# Phase 2 preparation closeout

10 October 2026. **Preparation complete; first GPU run awaiting approval.** No trained checkpoint, model accuracy or outer-assessment prediction was created.

## Prepared method

Fresh ImageNet B3, direct112×112, global pooling, two4096-unit ReLU layers, dropout0.5 and seven logits; full fine-tuning with Adam1e-4/batch16, balanced training-only augmentation/CE, BF16 training forward and FP32 validation. Exact source parameters and all implementation assumptions are in [recipe decisions](RECIPE_DECISIONS.md).

`r201_b3_dermai_adaptation_fold00_seed42` uses8111 fitting /902 inner-validation originals on fixed fold0. Training exposes5430 instances/class each epoch, **38,010 total /2,376 batches**. No outer assessment dataset exists in the runner.

Allowance50/minimum25, meaningful accuracy/F1 anchors0.002/0.003, patience10, LR reduction and settling, late flat-trend guard after30. Raw numerical winners are preserved independently. This replaces the earlier provisional30-epoch working plan.

## Verification evidence

- CPU pretrained forward/backward and real Adam update passed using four inner-training samples; all parameters are trainable, all gradients finite, both backbone/head changed. Temporary update state was discarded.
- Deterministic preprocessing checked on one inner-validation image without scoring it. Zero outer-assessment images opened; CUDA remained uninitialized.
- Full model: **33,801,775 parameters**, head23,105,543, backbone10,696,232. Optimizer coverage and checkpoint/optimizer/scheduler/RNG serialization round trip passed.
- Exact balanced exposure and original coverage, repeatable epoch sampling, wrong-partition rejection and meaningful-stopping controls passed.
- Independent synthetic CPU runner harness passed atomic reload, earliest raw-best tie selection, NaN-state/signature-drift rejection, probability/metric consistency, interrupted bookkeeping repair and complete checkpoint/PNG/PDF closeout. Synthetic fixtures were temporary and never entered experiment registries.
- Read-only implementation audit identified and resolved history LR/epoch-seed inconsistencies and missing strict determinism settings. Checkpoint recovery remains authoritative over CSV files; stale registry-lock handling is documented.
- Historical/V1 hashes, all588 historical master rows and127 checkpoint inventory entries remain preserved. New run registry remains empty.

Fresh ImageNet weight SHA256: `b3899882250c22946d0229d266049fcd133c169233530b36b9ffa7983988362f`.

CPU report / immutable config-code-protocol-weight launch signature: `results/image_level_replication/v2/preparation/b3_fold00_preflight.json` and `b3_fold00_launch_freeze.json`. These checks establish correctness; no future accuracy is estimated from them.

## Next boundary

[GPU proposal](PHASE2_PROPOSAL.md) contains the question, expected cost, exact start/monitor/resume commands and artifact paths. Separate approval is required. After an approved launch, confirm the log, initialized committed checkpoint and first accepted update, then stop interacting. No automatic next model, extra fold, refit, ensemble or outer inference is queued.
