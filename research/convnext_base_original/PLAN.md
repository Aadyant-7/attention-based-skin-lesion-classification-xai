# R203 — ConvNeXt-V2 Base on the original exploratory split

**Authorized:** prepare and launch one independent 30-epoch run. Stop interacting after logging, a recoverable checkpoint and the first accepted update are confirmed. No automatic next experiment, ensemble or test evaluation.

## Purpose and comparison

R202 achieved **93.2373% accuracy** (EMA, epoch 20) and **0.8886 best macro-F1** on its different 902-image V2 inner-validation cohort. It cannot fairly join the historical 94.011976% ensemble because the training/evaluation identities differ. R203 starts from external pretrained weights and uses the original exploratory manifest, enabling later prediction-level comparison and a bounded same-protocol ensemble study.

Research question: can this stronger ConvNeXt representation match or improve the original-split standalone models, and does it correct useful mistakes of the saved ensemble? A contribution is unproven until R203 finishes. This is a complete training-package experiment, not an isolated demonstration that model size, EMA or pretraining caused any eventual improvement.

## Frozen data and initialization

- Manifest: `data/splits/exploratory/image_level_dev_v1.csv`.
- SHA256: `75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb`.
- **7,009 training / 1,503 validation / 1,503 test** originals; approximately 70/15/15; seed 42. Only train and validation enter this runner.
- Class order: `akiec, bcc, bkl, df, mel, nv, vasc`.
- Train/validation image identities are distinct. The existing image-level development protocol shares 563 lesions; 596 validation images have a training lesion. This is exploratory validation, not lesion-independent or unseen-test performance.
- The original test identities remain excluded. No test images, probabilities, scores or labels are used for selection. Historical test outcomes already exist; R203 does not create a new independent-test claim.
- Native `convnextv2_base.fcmae_ft_in22k_in1k`, 87,699,975 parameters with seven-class head. FCMAE then ImageNet-22K/ImageNet-1K author pretrained weights; no R202 or other project checkpoint initialization.
- Local source: `.cache/torch/hub/checkpoints/convnextv2_base_22k_224_ema.pt`; SHA256 `dd7a88d111c8af2295cb744f5bba888eca26759984b1cac81804ef876807ce17`. This recorded digest checks local integrity.
- Architecture/weights references: [timm model card](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k), [author implementation](https://github.com/facebookresearch/ConvNeXt-V2). Their weights are CC BY-NC 4.0.

## Predeclared recipe

| Component | Setting |
|---|---|
| Duration | 30 planned epochs; no automatic extension, early rejection or next model; explicit manual stop finishes and commits the current epoch |
| Adaptation | Epochs 1–2 head only, frozen backbone in eval mode; epochs 3–5 unfreeze with backbone LR factors 1/3, 2/3, 1; then full fine-tuning |
| Head | Native average pool, LayerNorm2d, dropout 0.2, seven logits; native V2 GRN retained; no CBAM |
| Input | Whole-image RGB 224×224 bicubic resize; ImageNet normalization; no validation crop |
| Augmentation | Train-only independent horizontal/vertical flips 0.5 and uniform quarter-turn rotation |
| Sampling | Natural training originals once each per epoch; deterministic shuffle; final partial group retained |
| Imbalance/loss | Train-only square-root inverse-frequency weighted CE, normalized mean 1; FP32 loss sum divided by complete effective-batch target-weight sum |
| Optimizer | AdamW, betas 0.9/0.999, epsilon 1e-8, weight decay 1e-4; no decay on biases or one-dimensional normalization/GRN parameters |
| Learning rates | Head 1e-4; final backbone stage 3e-5; earlier stages multiplied by 0.8; stem ×0.4096 |
| Schedule | Epochs 6–30 cosine, progress `(epoch-5)/25`; final-stage LR 1e-7 at epoch 30; preserve group ratios |
| Batch | Microbatch 8 × accumulation 4 = effective 32; validation batch 8; two workers |
| Precision | BF16 forward; FP32 loss, softmax and validation; TF32 off; gradient clipping 1 |
| EMA | Initialize after head warmup; fixed decay 0.999 after accepted full-stage optimizer updates; FP32 on GPU; floating parameters averaged, buffers copied |
| Selection | Separate raw and EMA validation; maximum accuracy across branches, earliest epoch, raw wins same-epoch ties; macro-F1 selected independently |

Train class counts: `229,359,769,81,778,4693,100`. Recomputed weights: `1.1106702023,0.8870653949,0.6060935912,1.8674988896,0.6025777108,0.2453452106,1.6807490006`.

Relative to R202, the split, training-derived weights and 30-epoch cosine horizon change. The representation, optimizer, augmentation, effective batch, EMA and selection rules remain the same. The shorter horizon deliberately spends the available compute in the region where R202 obtained its useful results; it is not a guaranteed transfer of that learning curve. No oversampling, focal loss, Mixup, CutMix, TTA, multi-resolution or probability-weight search is added.

## Commands and locations

From `C:\Users\MARS\Desktop\College\Minor Project\Project Development`:

```powershell
.\.venv\Scripts\python.exe -B -m research.convnext_base_original.train --memory-check
.\.venv\Scripts\python.exe -B -m research.convnext_base_original.train --train
```

Manual monitoring from any PowerShell window:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\structured_experiments\convnext_base_original_v1\r203_convnextv2_base_original_exploratory_seed42\train.log' -Tail 20 -Wait
```

Resume after confirming the previous worker has exited:

```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
.\.venv\Scripts\python.exe -B -m research.convnext_base_original.train --resume
```

Safe manual stop:

```powershell
.\.venv\Scripts\python.exe -B -m research.convnext_base_original.train --request-stop --reason 'User requested safe stopping'
```

An explicit manual stop requires new authorization and `--allow-user-stop-resume`; interruption recovery uses plain `--resume`. Recover from committed `latest.pt`, preserving the recipe, history, LR, optimizer, EMA and RNG. Interrupted CSV/figure bookkeeping can be repaired from saved predictions; partial uncommitted epochs replay from the last complete checkpoint.

- Config/source: `research/convnext_base_original/`.
- Results/log: `results/structured_experiments/convnext_base_original_v1/r203_convnextv2_base_original_exploratory_seed42/`.
- Checkpoints: `checkpoints/structured_experiments/convnext_base_original_v1/r203_convnextv2_base_original_exploratory_seed42/`.
- Preparation evidence/freeze: `results/structured_experiments/convnext_base_original_v1/preparation/`.

Save latest/final states, best accuracy and macro-F1 plus all raw/EMA branch winners; exact checkpoint hashes; accepted-update and numerical-event counts; stopping reason; train/validation/LR history; validation predictions/probabilities; class-wise scores; confusion matrices; PNG/PDF curves/comparison figures; local and master registry entries. Historical R202 and ensemble artifacts remain intact.

Estimated runtime: **roughly 1.5–2 hours** on RTX 4060 8GB, extrapolated from R202's measured 50-epoch run and the smaller 7,009-image training cohort. R202's identical native architecture/batch/EMA synthetic GPU preflight peaked at 3.00 GiB training; R203 must pass its own frozen preflight. These are provisional estimates, not accuracy guarantees. Keep the launch preflight's disk reserve available for atomic checkpoints.

After the user returns, close out R203 first, inspect errors against the old ensemble on the identical 1,503-image validation cohort, and consider only a small bounded CPU soft-voting comparison. No historical test score is used for those decisions, and no test inference or subsequent model launches automatically.
