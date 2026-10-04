# High-accuracy literature gap analysis â€” 4 October 2026

Scope: existing verified review, SciSpace screening (50 entries/49 distinct DOI candidates), refreshed primary papers and empirical S02â€“S18. Workbook and historical evidence unchanged. Missing extraction fields mean **unverified**, not absent. Keep the base paper and foundational HAM10000 references in research/literature/review.csv and references.bib. No paper score is an expected result for our cohort.

## Primary high-performance evidence

### Sravani & Koppu (2026)
[Primary full text](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2026.1847649/full), Â§Â§3.3â€“3.7, Tables3â€“6.

| Field | Reported detail |
|---|---|
| Models/pretraining | ImageNet B3/ResNet101/DenseNet201; all layers fine-tuned |
| Split/train fraction | Lesion80/20 nominal8012/2003; metrics on1103single-image lesions; cohort construction ambiguity |
| Resolution |224; dataset-specific normalization |
| Balancing/loss | Inverse-frequency weights, manual multipliers/clamp; focal Î³2.2 |
| Augmentation | Class-specific spatial/photometric transforms, erasing; Mixup p.3 Î±.2; CutMix unverified |
| Optimizer/LR | AdamW; B3 1e-4, Dense7.5e-5, Res5e-5 |
| Scheduler/regularization | StepLR .7/10epochs; decay.0015; clip.5; dropout.65/.55/.45 custom head |
| Budget/patience |50maximum; patience12/minimum25; seed10; AMP |
| Ensemble/selection | Soft voting .4B3/.4Dense/.2Res; empirically validation-based weights/LRs; maximum validation accuracy |
| Evaluation |96.37%; weightedF1 .9623; reported3fold95.56Â±.32; not our macroF1/cohort |
| Unverified details | Exact weight versions, schedule of progressive unfreezing, external untuned test selection |

These differences motivate hypotheses, not copying scores or manual class/ensemble weights.

### MedFusionNet (2025)
[Primary paper](https://www.nature.com/articles/s41598-025-31816-2), methods and training sections.

| Field | Verified detail / limitation |
|---|---|
| Models | ConvNeXt/ViT attention-based feature fusion; exact checkpoint versions unverified here |
| Preprocessing |224px/ImageNet normalization; flips, rotations, color augmentation |
| Balancing/loss | SMOTE described; categorical CE; focal/Mixup/CutMix unverified |
| Optimization | Adam1e-4 best configuration; decay formula stated; batch16;100epochs |
| Evaluation | HAM98.8%; protocol/train fraction/lesion grouping not clearly established in retrieved methods |
| Ensemble | Learned feature fusion, not our fixed probability average |
| Unverified fields | Dropout, weight decay, patience, progressive unfreezing, independent test/weight-selection details |

Supports representation diversity and regularization questions, not direct comparison or an automatic Swin run.

### DermAI1.0 (2023)
[Primary indexed text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10573070/) and [publisher PDF](https://mdpi-res.com/d_attachment/diagnostics/diagnostics-13-03159/article_deploy/diagnostics-13-03159.pdf). Search-indexed primary excerpt corroborates ResNet101/InceptionV3/B3/B7/Dense201 transfer learning and softmax/weighted/stack ensembles. Earlier review records B3 97.01% and augmented/K10 evaluation. Browser full-text access remains restricted; no new verification of train fraction, resolution, balancing/loss, Mixup/CutMix, optimizer/LR/scheduler, dropout/decay, patience or weight-selection protocol. Do not treat its score as equivalent to our split.

## Audit of broader input and counter-evidence

SciSpace decisions:8priority,22candidate,10different-task,8deferred,1duplicate,1survey; most claims remain extraction-only. SynthraXCoreNet is a broad ensemble hypothesis, not verified protocol evidence. ML-IGIA has documented six/seven-class and label-dependent-weight ambiguities. Patch-attention reference reports mean sensitivity, not accuracy. Retain these distinctions and all base references.

A primary EfficientNet comparison also reports B4 only87.91%/F1 .87: [original article](https://www.sciencedirect.com/science/article/pii/S2772528621000340). This counter-example reinforces that scale/family alone cannot promise95%.

## What differs most from our ~92.5% system?

Ranked hypotheses, not measured causal effect sizes:

1. **Protocol and reporting differences.** We use1503natural validation images with563shared train/val lesions (596valimages affected). Papers may use different train fractions, single-image lesion subsets, folds, augmented cohorts or weighted rather than macroF1. These differences can dominate headlines. Never modify our split to inflate accuracy.
2. **Generalization/regularization package.** Our strong models fit training nearly perfectly while validation can deteriorate. Verified papers use richer spatial/minority augmentation, Mixup and stronger head regularization. Could matter more than more epochs; local historical augmentation/focal packages do not isolate positive causal effects.
3. **Missing scaled/residual representation families.** Two final controls, B3 and ResNet101, answer a meaningful diversity question. DenseNet failed the material-gain gate; ConvNeXt-Small gave modest accuracy gain with lower macroF1. Swin lacks sufficiently strong verified same-task evidence to justify a third run now.
4. **Ensemble member quality/complementarity.** Our largest gains came from strong complementary members. Bounded equal voting, then two frozen weight templates on one set, may improve balance; no continuous weight optimization or oracle routing.
5. **Optimization budget/scheduler.** Peaks often12â€“18; current models commonly declined afterward. A50epoch allowance with sensible patience is a future strict recipe, not evidence that20was insufficient. Do not select the final epoch by default.
6. **Resolution/attention/XAI.** Useful scientific questions, but TTA failed and CBAM was mixed. No new resolution or attention runs in this final block. XAI remains essential after classifier freeze.

## Final bounded batch policy

Only EfficientNet-B3 (S16, existing unlaunched config) and ResNet101 (S19, ImageNet1K V2). Same224/exploratory cohort/seed42/common weightedCE/full fine-tuning/AdamW discriminativeLR policy/20cap/patience5. No epoch8gate. Different pretrained packages are disclosed; no architecture will be retrained after this decision block.

Resource probes measured real training batches; short timing is not full epoch throughput. Sequential scheduling chosen because concurrent dataloaders/checkpoint states lack comfortable host-RAM headroom, despite GPU fit. No two-heavy-job test under unsafe host pressure.

Freeze member sets before equal fusion results, and freeze two non-equal templates before evaluating them. Maximum6equal+2weighted candidates; no grid, class-specific weights or calibrator. Material success requires >=1pp accuracy with non-decreasing macroF1, OR >=.01 macroF1 with non-decreasing accuracy, versus FP32reference. Protect akiec/melanoma/df/vasc recalls from substantial loss. Otherwise freeze the existing reference. This gate is a decision heuristic, not a significance claim.

See autonomous launcher research/run_final_architecture_selection.py. The generated FINAL_ARCHITECTURE_SELECTION.md closes architecture search permanently; strict fresh training/test/XAI remain unexecuted.
