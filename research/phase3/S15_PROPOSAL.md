# S15 proposal: DenseNet201 strong-backbone screening

**Prepared; wait for human GPU approval. S14 is deferred.** [Bounded high-performance plan](HIGH_PERFORMANCE_ENSEMBLE_PLAN.md) selects DenseNet201 and B3 plus one conditional Swin-T slot; no full architecture sweep.

## Research question / reason for first position

Can fully fine-tuned DenseNet201 achieve competitive standalone exploratory performance and recover errors that ConvNeXt/S12 miss, enabling a materially stronger heterogeneous ensemble?

Dense concatenation/reuse is a missing family in the current ConvNeXt/B0/V2-S ensemble. The focused [literature update](../literature/HIGH_PERFORMANCE_ENSEMBLE_EVIDENCE.md) verifies recent DenseNet/B3/residual ensemble methods and explains why their cohorts/metrics/training differ. B3 sometimes ranks higher in those papers, but DenseNet is the more distinct representation test first. DenseNet's old ImageNet score is not a predictor of HAM10000 accuracy. No promise of95% or of beating ConvNeXt standalone.

## Exact recipe

- Model `densenet201`, weights `DenseNet201_Weights.IMAGENET1K_V1`; no added CBAM. Native dense feature stack and final ReLU retained. Seven-class parameters18,106,375.
- Replace1000-class head with GAP/dropout.2/linear7; full end-to-end fine-tuning from epoch1, no permanent freeze or training from scratch.
- Existing `exploratory_screening_v1`: seed42,224square bilinear direct resize/ImageNet normalization,horizontal flip.5, train-only sqrt inverse-frequency weights, weighted cross-entropy, AdamW3e-5backbone/1e-4head, weight_decay1e-4, scheduler ReduceLROnPlateau accuracy,.5factor/2patience.
- Microbatch16/accumulation2 →effective32, AMP training/GradScaler,20-epoch maximum/five-stale-accuracy early stop. Primary best accuracy, earliest tie; independent macro-F1 winner.
- Same7009train/1503validation/1503originaltest exploratory manifest. No test loader. Shared train/val lesions disclosed; later strict confirmation postponed.
- Keep successful ConvNeXt recipe for this backbone question. Historical focal/strong augmentation did not reliably help; no evidence supports making those default now. A literature-inspired training package remains possible only if new-class/curve evidence justifies it.
- Existing finite-output validation diagnostics apply; unapproved numerical policy changes do not happen silently.

## Resources

RTX4060 8GB; **approximately25–45min /3–6GB peak allocated VRAM**, unbenchmarked. Dense concatenation can cost more memory/kernel time than parameter count suggests. CPU interface checks do not establish GPU fit. If it cannot fit, stop and propose an explicit microbatch/accumulation amendment; effective32 should be retained. Allow roughly2GB local checkpoint/atomic-save space plus77.4MB pretrained download if uncached.

## Start / monitoring / resume

After approval, from project root:

```powershell
.\.venv\Scripts\python.exe -u -m research.train --config research/configs/phase3/s15_densenet201_none_exploratory_seed42.json
```

Manual monitor:

```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
Get-Content .\results\structured_experiments\s15_densenet201_none_exploratory_seed42\train.log -Tail 30 -Wait
```

Interrupted run: same start command plus `--resume`, restores the latest committed epoch model/optimizer/scheduler/scaler/RNG/history. Source/config/version guards require the same reviewed runner and config. No restart or silent epoch/config change.

## Saved artifacts and later ensemble

- Config: `research/configs/phase3/s15_densenet201_none_exploratory_seed42.json`.
- Results/log: `results/structured_experiments/s15_densenet201_none_exploratory_seed42/`, `train.log`.
- Checkpoints: `checkpoints/structured/s15_densenet201_none_exploratory_seed42/` →`best.pt`, `best_macro_f1.pt`, `latest.pt`.
- Histories, metrics, class scores, primary/secondary selected predictions/probabilities, matrices/PNG300dpi+PDF figures, curves, config/environment/pretraining/source hashes, progress and master registry. Closeout verifies/generates secondary matrices/class plots and comparison/error figures.
- Comparison review destination: `results/model_comparison/structured/s15_backbone_review/` (created during closeout, not claimed existing now).
- First candidate if admitted: equal ConvNeXt+DenseNet. Then ConvNeXt+DenseNet+B3; optional4members only after complementarity justifies it. Existing V2-S/B0 remain available but no exhaustive subsets. Equal voting first, then at most two fixed global weight templates on one selected set, all disclosed; exact future configurations frozen before fusion evaluation.

CPU readiness: two synthetic interface/gradient/config tests and both candidate metadata preflights passed.224px seven-class forwards and64px synthetic backward passes finite; gradients reach the backbone; no optimizer training, real-image inference, weights download, CUDA query or test images. S15/S16 have no result/checkpoint directories or new registry rows yet. Source additions preserve existing adapters/checkpoints; frozen historical launch code remains authenticated from Git.

Approval covers S15 only when explicitly given. Independent launch→confirm log and initial atomic checkpoint creation→manual monitor/resume/paths→stop immediately until user returns. Do not launch S16, S14, fusion or any other experiment automatically.
