# S10 numerical failure diagnosis and proposed recovery

**Prepared only; human approval required before GPU resumption.** No training, optimizer step or test-image inference was run during diagnosis. S10 remains one failed/incomplete experiment, not a new result or restarted run.

## Confirmed evidence

- All 219 effective-batch steps of epoch14 reached validation with finite checked training loss. The crash was the aggregate validation-loss check; epoch14 was never committed. **Latest valid checkpoint is epoch13**, with 88.6228% accuracy, .814538 macro-F1 and 1.277999 weighted validation loss. Both selected winners also point to13.
- Original best/latest/F1 checkpoints load on CPU; model parameters, BatchNorm buffers and optimizer tensors are finite. Config, source hashes, runtime, history and winner model states match. Scaler16384 is a valid state, not evidence of corruption.
- All1,503 validation images decoded on CPU with finite normalized inputs and valid labels; complete epoch13 FP32 inference/loss/probabilities are finite. CPU FP32 accuracy88.4897% differs by two predictions from saved CUDA AMP; this comparison changes both device and precision and is diagnostic only, not a replacement experiment result.
- An outlier `ISIC_0025851` produced absolute FP32 logits597.83 (most highest-logit images ~12–13). Its batch had intermediate activations15,849 in FP32 and17,136 in CPU FP16, both finite. The targeted FP16 probe **did not reproduce overflow**. This supports examining activation amplification/eval BatchNorm and CUDA AMP; it does not identify the epoch14 cause or justify removing the sample.
- Loss already casts logits to FP32 before stable weighted cross-entropy. It is not `log(softmax)` and its positive class-weight denominator cannot be zero. Merely casting loss to FP32 is therefore not a fix. With finite FP16 logits/weights, this small FP32 sum is far below overflow range; nonfinite model outputs are the leading failure path, but the originating operator was not logged.

**Exact original cause remains unconfirmed.** Epoch14 logits/probabilities, failing sample IDs and model state were not saved. Original NaN vs Inf and first layer cannot be established retrospectively. Current checks find no saved-checkpoint corruption or validation decode problem; they cannot inspect the lost epoch14 state. CUDA-only overflow, nonfinite eval buffers after training, or a loss/kernel failure must be distinguished by an approved deterministic replay. [PyTorch2.11 AMP documentation](https://docs.pytorch.org/docs/2.11/amp.html) explains FP16 range and FP32 loss policies; it is mechanism context, not proof that this particular run overflowed.

## Prepared guarded recovery

`research/train.py` now checks inputs/weights/targets, logits, loss and probabilities **per batch before metrics**. On failure it saves sample IDs, NaN/Inf counts, first nonfinite module on same-batch replay, model/optimizer finiteness, and a diagnostic model/optimizer/scheduler/scaler/RNG snapshot. A diagnostic snapshot is never accepted as `latest.pt`.

Only the reviewed S10 amendment may enable recovery. If AMP fails with finite model/optimizer state and the identical batch succeeds in FP32, **repeat the entire validation pass in FP32**, retaining every sample. Otherwise stop and preserve the previous committed checkpoint. No clipping, `nan_to_num`, sample exclusion, BN reset, LR/batch/input change, or fabricated loss. Training continues in the original AMP policy.

`research/phase3/s10_resume_fix.json` pins epoch13, original latest checkpoint SHA, unchanged config SHA, original/new source maps and this policy. Only `research/train.py` may migrate; ordinary resume config/runtime/source guards remain enforced. New committed checkpoints carry the amendment for subsequent ordinary `--resume`. Eighteen CPU runner/recovery tests passed, including overflow recovery, corrupted-weight refusal and migration rejection. Actual epoch14 CUDA recovery is **not yet verified**.

## Comparability

The split, sample population, labels, architecture, training precision, optimizer/scheduler/scaler/RNG, LR, batch size and epoch budget remain unchanged. Old epochs and original artifacts remain preserved. Normal finite AMP validation is unchanged; any FP32 recovery is explicitly recorded in history, diagnostics, amendment and summary. Precision can change probabilities/predictions and therefore selection/scheduler decisions. Treat it as a disclosed numerical implementation amendment, not perfectly identical evaluation; do not claim a gain caused solely by architecture or hide this change. No new recipe/ID or accuracy experiment is created.

Resume boundary: `checkpoints/structured/s10_efficientnet_v2_s_none_exploratory_seed42/latest.pt`, SHA `fc9556322f6ab01da9ae192e33d678a28039a378bd2e0fb85bea80123978c2dd`. Replay epoch14, then at most seven remaining epochs subject to original early stopping. Model, optimizer, scheduler, scaler, RNG and both winner states restored; no new pretrained initialization/download.

After approval, from the project root:

```powershell
.\.venv\Scripts\python.exe -u -m research.train --config research/configs/phase3/s10_efficientnet_v2_s_none_exploratory_seed42.json --resume --resume-fix research/phase3/s10_resume_fix.json
```

Manual monitoring:

```powershell
Get-Content .\results\structured_experiments\s10_efficientnet_v2_s_none_exploratory_seed42\train.log -Tail 30 -Wait
```

Existing result/checkpoint folders remain unchanged. Failure diagnostic JSON/PT files will be added to the result folder if needed. Allow several additional GB for forensic snapshots. Launch independently; confirm resumed logging and a new atomic checkpoint beyond13, then **STOP immediately**, no epoch polling/completion wait. User returns for closeout. If replay also fails, stop with diagnostic evidence and retain epoch13; do not automatically run another experiment.

Frozen originals: `results/audit/s10_nonfinite_20261003/preserved/{checkpoints,run_artifacts}/`. `preservation.json` verifies originals/copies; `diagnosis.json` stores CPU checks and targeted precision probes; `recovery_preflight.json` records no launch and passed gates. PT/log copies remain local/ignored. Existing registry row remains failed until approved recovery.

Current Phase Progress: 87%
Total Project Progress: 72%
