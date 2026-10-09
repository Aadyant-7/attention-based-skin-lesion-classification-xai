# Final CBAM reporting and post-development audit — 9 October 2026

User authorized resuming the reporting/XAI phase and evaluating the finalized CBAM-inclusive five-model ensemble for the record. This supersedes the earlier prohibition on any further test inference **only for this one fixed S80 audit**. It does not authorize training, a new model/weight/checkpoint search, or another independent-test claim.

## Fixed method

Use S80 unchanged: S79 ConvNeXt-Tiny+CBAM accuracy checkpoint33; S18 ConvNeXt-Small macro-F1 checkpoint17; S15 DenseNet201 macro-F1 checkpoint14; S10 EfficientNetV2-S macro-F1 checkpoint15; S03 EfficientNet-B0 accuracy checkpoint20. Each contributes0.2. FP32, no TF32, identity only, square224 bilinear antialias, RGB/ImageNet normalization, class order akiec,bcc,bkl,df,mel,nv,vasc. No TTA, extra resolution, metadata, inference augmentation or fitting. Do not choose S79's F1 checkpoint after seeing test scores.

## Evaluation contract

- Verify frozen checkpoints/configs/code, the same original1503 test identities, zero test image/lesion overlap with every member's training/development cohort, and preserve the complete original S31 test artifact tree by hash.
- Generate each member's test probabilities once, sequentially, with labels withheld from the inference dataset. Equal average, then score the saved probabilities. Constituents are reported descriptively from these same passes; no alternative test ensemble.
- Exclusive execution receipt blocks accidental repeated inference. On failure preserve evidence; finish reporting from completed saved passes where possible, rather than silently rerunning inference.
- Original S31 first test:86.7598% /0.794473. S80 validation:93.3466% /0.877017. These use different models/training protocols; report comparison descriptively.
- This cohort is image/lesion-disjoint from model training, but its earlier outcome is already known and the project continued development afterward. Therefore this is a **previously evaluated held-out cohort / post-development test audit**, not a pristine first-and-only evaluation. Do not overwrite or retroactively revise the original report.
- Save each member and ensemble metrics, probabilities/predictions, class scores, raw/normalized confusion matrices and PNG/PDF figures; update the registry with explicit post-development provenance. Test score will not trigger changes to S80.

## XAI contract

Use only exploratory validation cases, deterministically selected by first image ID within correct/incorrect melanoma and akiec categories and representative remaining classes. Up to11 distinct cases; absent categories disclosed. Explain the saved S80 predicted class using Grad-CAM of each branch's actual0.2-weighted class probability. Execute branches sequentially; this equals the branch derivative of the summed ensemble probability and avoids holding all five graphs simultaneously. Verify recomputed probabilities/predicted labels against saved S80 outputs.

Targets: ConvNeXt final feature stage; DenseNet denseblock4; EfficientNet final feature layer. Preserve raw/normalized maps, overlays, panels, label/confidence metadata and actual S79 channel/spatial sigmoid attention weights. Equal-average normalized CAM composite is a display summary, not an exact causal attribution or diagnostic segmentation. Grad-CAM/CBAM do not change predictions or establish clinical validity.

## Reporting

Artifacts: `results/final_cbam_reporting/v1/`; code: `research/final_cbam_reporting/`. Checkpoints remain in their existing folders; no rewriting or training. Generate a focused final reporting summary and figure index, not a new complete manuscript before the user's remaining report requirements arrive. Highlight actual validation accuracy with the protocol label visible. Distinguish retained non-CBAM S53(93.6128%), attention-inclusive S80(93.3466%), and highest observed exploratory variants with their different input contracts/tradeoffs; do not pool their claims.

GPU work is inference/XAI only; batch16, one model resident at a time. Expect several minutes, not another multi-epoch training session. Manual monitor:

```powershell
Get-Content 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\final_cbam_reporting\v1\audit.log' -Tail 25 -Wait
```

Commands (from project root; prepare before the single inference run):

```powershell
.\.venv\Scripts\python.exe -B -u -m research.final_cbam_reporting.finalize --prepare
.\.venv\Scripts\python.exe -B -u -m research.final_cbam_reporting.finalize --evaluate
.\.venv\Scripts\python.exe -B -u -m research.final_cbam_reporting.finalize --xai
```
