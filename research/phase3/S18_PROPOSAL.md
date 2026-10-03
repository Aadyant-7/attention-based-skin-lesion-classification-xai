# S18: staged ConvNeXt-Small proposal

Not approved or launched. Replaces the automatic DenseNet/B3/Swin sequence. S16 B3 remains prepared but deferred; S14 remains deferred. S17 identifies the single approved CPU fusion, so S18 is the next training ID.

## Why this model
Research question: does scaling the proven ConvNeXt family materially improve standalone performance and its contribution to our heterogeneous ensemble?

Our Tiny already reached 91.75%; Small increases depth/capacity without changing the family or preprocessing. Installed torchvision 0.26 metadata: original ImageNet heads Tiny 28.59M parameters/82.52% top-1/4.456 GFLOPs; Small 50.22M/83.616%/8.684 GFLOPs. Our seven-class Small adapter has 49.46M parameters and preserves native LayerNorm. These facts motivate a pilot, not a predicted HAM10000 score.

[Small official documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_small.html), [Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html), [original ConvNeXt paper](https://arxiv.org/abs/2201.03545).

| Candidate | Decision |
|---|---|
| ConvNeXt-Small | First choice: strongest direct local evidence of approaching Tiny, larger capacity. Same-family error correlation is a risk; consider replacing Tiny, not assume diversity. |
| EfficientNet-B3/B4 | Existing verified HAM ensemble papers support these families, but protocols differ. B3 is smaller than Tiny; B4's usual 380px adds resolution/cost differences. Our B0/V2-S evidence is weaker than Tiny. Defer. |
| Swin-T | Useful attention diversity, but less verified standalone HAM evidence for our recipe; no local success yet. Reserve rather than automatically train. |

Working success target: roughly 91.75-93% standalone, not a forecast or confidence interval. With one Tiny run, a calibrated Small outcome range cannot be estimated; below 91% remains possible. No supported standalone 95% expectation.

## Matched recipe
ImageNet pretrained ConvNeXt_Small_Weights.IMAGENET1K_V1; full fine-tuning; same seed42, exploratory manifest, 224px, horizontal flip, ImageNet normalization, training-only weighted CE, AdamW learning rates, effective batch32 (micro16/accumulation2), AMP and patience5 as S06. No CBAM or simultaneous training change. Locked test never loaded.

## Predeclared review gates
Config retains max_epochs20. Runtime boundaries pause after committed epochs; optimizer/scheduler/scaler/RNG/config remain unchanged. Pilot figures/probabilities are saved under screening/epoch_08 (and optionally epoch_12). No automatic continuation. Stop on patience also remains active. Keep negative pilots in the registry and paper.

1. Train only through epoch8, then exit for review. Tiny was 90.09%/.8355 at8; requiring its final 91.75% too early would reject our proven model.
2. Continue only if best accuracy >=90.0%, best macro-F1 >=.83, and best accuracy in epochs6-8 improves >=.5 percentage points over best epochs1-5; alternatively already >=91.75%/.86. Failing the gate stops this candidate. These compute heuristics can reject a later improver; they are not validated learning-curve predictors.
3. For a promising pilot below 91.75%, approve only continuation to12, then pause. Full continuation requires best accuracy >=91.25%, macro-F1 >=.85 and >=.3pp accuracy improvement over the epoch8 best; alternatively already >=91.75%/.86. Review minority-class collapse before approval.
4. If epoch8 already meets 91.75%/.86, propose full continuation, still awaiting approval. Full run retains cap20/patience5. Freeze gates before launch; do not tune them after seeing pilot results.

## Cost and commands (approval required)
Unbenchmarked RTX4060 8GB estimate: pilot8 epochs 10-15min; total20 epochs 25-40min; allocated VRAM roughly3-5GB at micro16. Based on Tiny14.94min and roughly doubled operations, not measured Small throughput. Reserve about4GB checkpoint/atomic-write space. If OOM, stop and request an explicit microbatch amendment; no silent change.

From project root, pilot:
```powershell
.\.venv\Scripts\python.exe -u -m research.train --config research/configs/phase3/s18_convnext_small_none_exploratory_seed42.json --stop-after-epoch 8
```
Manual monitor:
```powershell
Get-Content .\results\structured_experiments\s18_convnext_small_none_exploratory_seed42\train.log -Tail 30 -Wait
```
Only after pilot review/approval:
```powershell
.\.venv\Scripts\python.exe -u -m research.train --config research/configs/phase3/s18_convnext_small_none_exploratory_seed42.json --resume --stop-after-epoch 12
```
Only after full continuation approval:
```powershell
.\.venv\Scripts\python.exe -u -m research.train --config research/configs/phase3/s18_convnext_small_none_exploratory_seed42.json --resume
```
Interrupted pilot recovery uses --resume --stop-after-epoch 8 before8 is committed; after8, wait for review instead of repeating the boundary. Source/config resume guards remain enabled.

Results/log: results/structured_experiments/s18_convnext_small_none_exploratory_seed42/; pilot artifacts screening/epoch_08/, optionally epoch_12/.
Checkpoints: checkpoints/structured/s18_convnext_small_none_exploratory_seed42/ (best.pt, latest.pt, best_macro_f1.pt).
Master registry: results/master_experiment_registry.csv. Standard final metrics, probabilities, matrices/class scores and curves publish on completion.

If retained, inspect error overlap with Tiny/S10/B0; test one justified ensemble replacing Tiny in S12. Only pair Small/Tiny if actual error diversity supports it. Launch-and-stop policy: confirm first saved checkpoint/log, stop interacting, user returns at the automatic screening boundary.
