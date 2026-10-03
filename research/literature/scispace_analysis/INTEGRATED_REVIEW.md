# Integrated literature review — 3 October 2026

Review snapshot prepared before S01 training. Current results/strategy are in [S01 closeout](../../phase3/S01_CLOSEOUT.md) and [two-stage amendment](../../phase3/TWO_STAGE_STRATEGY.md); its proposed all-strict screening sequence is superseded. Source extraction/screening decisions remain unchanged.

## Evidence and screening

Latest input: `research/literature/incoming/SciSpace Literature Review.xlsx`, sheet `table1`, **A1:H51**, SHA-256 `1cf7ed7cad73ff48c235d4364509f95c52a86c3a9a23512c571fa2c31278c8db`. It contains 50 entries, eight columns, and no formula cells. The workbook was read without edits. `screening.csv` gives a decision, source coordinates, method/result extraction, rationale and verification scope for **every entry**. `provenance.json` records the reproducible counts; `../screen_scispace.py` produces these using bundled Python/openpyxl. Full extracted source text remains local in `.cache/research_literature/scispace_extracted.json`.

Rows 26 and27 have the same AWB-loss DOI after case normalization: **49 distinct DOI candidates**, not 50 independent studies. Screening decisions: 8 priority references, 22 candidates requiring further protocol checks, 8 deferred topics, 10 different-task context entries, 1 survey, 1 duplicate. These are relevance decisions, not quality scores. A deferred paper remains indexed and may become useful for a later question.

Six representative primary sources were checked for selected fields; most workbook claims remain extraction-only. Unavailable full texts and missing split details remain unverified. The Excel includes UI fragments in paper identifiers and generic inferred gaps. Its “no limitations” statements cannot establish absence of limitations. Row3 directly contradicts its paper's explicit limitations section. Dataset spellings such as “HAM1000” sometimes also appear in source abstracts, so they are not automatically SciSpace errors. Row 46's10,050 count conflicts with the foundational 10,015 count; row 18 describes ISIC2357, not our HAM10000 population.

## Recurring approaches and implications

Keyword screening across extracted methods/gaps/results/contributions/challenges of the 49 distinct entries finds imbalance 27, ensemble/fusion 25, preprocessing/augmentation 20, attention 18, transfer learning 15, XAI 12 and transformers 6. **These are text mentions, not counts of verified implementations, successes or independent effect sizes.** The pattern supports questions to test; it cannot rank architectures by averaging their reported accuracies.

| Theme | Representative workbook rows | What to retain for our experiments |
|---|---|---|
| Transfer learning and architecture comparisons | 5,17,18,34,44,47,51 | Exact ImageNet weights; common head/input/selection; fine-tune a pretrained CNN rather than train from scratch. AlexNet/Xception/large EfficientNet are context, not compulsory runs. |
| Attention | 3,6,16,19,24,41,50 | Channel/spatial/soft/patch/class-wise attention recur. First isolate **added CBAM** on a matched backbone; existing EfficientNet/MobileNet SE remains native. Patch/class-wise approaches are possible later, not interchangeable with CBAM. |
| Imbalance | 6,17,26/27,34,38,41,42,47 | Weighted losses, oversampling, balanced sampling and smoothing need separate controls. Compute weights/resampling from training only; retain natural validation support and inspect melanoma/minority recall. |
| Ensemble and feature fusion | 2,3,4,9,12,13,14,23,38,43,44,46,48,49 | Diversity may help but is not guaranteed: row 44 reports a weaker ensemble than its best individual model. Start with two complementary models and fixed equal probability fusion; save per-image probabilities for disagreement/error analysis. |
| Preprocessing and augmentation | 6,11,18,32,38,39,40,41,42,46,48 | Geometry/colour changes, hair removal, segmentation, wavelets, synthetic samples and TTA are distinct interventions. Do not introduce several simultaneously. Segmentation needs masks/provenance; hair removal can remove useful signal. |
| XAI and calibration | 3,4,9,11,12,19,24,28,30,37 | Grad-CAM, SHAP, LIME, Integrated Gradients and attention maps answer explanation questions. Grad-CAM is our first selected-model explanation, including errors; maps do not validate clinical correctness or improve accuracy by themselves. Calibration is a later separate objective. |
| More expensive/separate questions | 8,15,21,22,31,33,37,48 | Contrastive/MAE/ViT, distillation, multimodal/federated learning and GAN synthesis need additional training, inputs or supervision. Keep indexed; do not spend our first run budget here. |

## Selected primary-source checks

The following statements are verified only to the scope stated; author-reported results have not been reproduced.

- **Gessert patch attention (row 6):** author abstract supports high-resolution patch attention, pretrained CNNs and comparison of imbalance strategies. Its reported 7% and 3% improvements concern **mean sensitivity**, not ordinary accuracy. Full evaluation protocol is not verified here. [Author preprint](https://arxiv.org/abs/1905.02793), [publication metadata](https://pubmed.ncbi.nlm.nih.gov/31071016/).
- **ML-IGIA (row 3):** paper describes lesion-ID-based preprocessing,70/15/15 partitions and training-only augmentation; it reports 94.52%. Tables16/17 call the comparison “six classes” despite a seven-class description. IGIA uses correctness labels to derive weights, so the partition used to fit weights must be established before independent-test interpretation. Its limitations include one dataset and many classifiers, contradicting Excel's absence claim. This motivates bounded fusion, not immediate reproduction or an accusation of leakage. [Primary article](https://journals.sagepub.com/doi/10.1177/20552076241312936), preprocessing, IGIA Steps1–4, Tables16/17, Threats to validity.
- **Hu feature fusion/loss (row 42):** publisher abstract confirms multi-scale fusion, weighting/smoothing/resampling and separate hair-removed/segmented datasets; reported HAM10000 ACC94.0%, AUC99.3%. Abstract does not establish our split or an isolated gain from each intervention. [Publisher abstract](https://www.sciencedirect.com/science/article/pii/S0010482524006796).
- **Fraiwan/Faouri (row 47):** primary abstract describes 13 transfer models and best overall 82.9%, showing that high nineties are not universal for this task. Figure captions include70/30 and80/20 splits; lesion separation remains unverified. [Primary abstract and captions](https://pubmed.ncbi.nlm.nih.gov/35808463/).
- **Progressive class-wise attention (row 50):** author abstract confirms class-wise multi-scale attention and reported 97.40% HAM10000 /94.9% ISIC2019. No comparable lesion-disjoint protocol is established by the abstract. [Author preprint v1](https://arxiv.org/abs/2306.07300v1).
- **DCENSnet (row 2):** indexed publisher abstract supports three customized CNNs/dropout and reported 99.53%; full text retrieval failed. Split, balancing sequence and averaging remain unresolved. [Publisher source](https://www.sciencedirect.com/science/article/pii/S1746809423011904).

## Evaluation differences that prevent a pooled leaderboard

Before using a numeric result in the paper, record image/lesion/patient grouping; duplicate removal; class count; original/subset/augmented support; balancing sequence; train/validation/test boundaries; checkpoint/ensemble tuning partition; cross-validation design; pretraining overlap; metric units and averaging. A label-dependent ensemble fitted on evaluation labels cannot be interpreted as independently evaluated. Absence of reported grouping is **uncertainty**, not proof of leakage.

Examples requiring separation: row 15 reports kappa; row 29 AP/AUC; row 6 mean sensitivity; rows10/25 segmentation Dice/IoU/pixel accuracy; rows24/28 binary classification; row 9 weighted P/R/F1; row 36 validation accuracy; row 37 mixes segmentation accuracy and roughly 85% classification. Row 38 contrasts original 86% with balanced 98%; row 41 contrasts unbalanced 69.75% test with SMOTE 95.94%. These extracted contrasts require split/support/balancing verification before attributing gains to learning. They do not justify balancing our validation/test sets.

## Integrating the existing verified and historical evidence

The 12 curated primary references in `../sources.json`, their `../review.csv`, `../references.bib` and `../evidence_notes.md` remain part of the review. This supplement expands their scope without replacing them or silently promoting every Excel claim to verified evidence.

- **HAM10000/Tschandl2018** remains the dataset authority:10,015 images, repeated views of lesions, seven classes. Our frozen strict manifest has 7,009 train/1,503 validation/1,503 test, zero lesion intersections, SHA-256 `db1ce9f8b83f176001dd89fd332fb1668c7d2ba5f58d77157d293a09e2c2e696`. Lesion separation does not prove patient separation. Existing dataset tables/audits live in `results/datasets/`.
- **Pardede2026 base paper** remains the motivation and lightweight B0/MobileNet comparison. Its98.86/98.88% reported accuracy, conflicting balancing descriptions and 7,041 test support are already audited. That is not our1,503-image natural strict evaluation population. See `docs/base_paper_protocol_audit.md`; no claim that the unpublished pipeline has been reconstructed.
- **Datta soft attention**, **Gessert multi-resolution ensembles**, original **CBAM/EfficientNet/MobileNet/ResNet/DenseNet/ConvNeXt/Grad-CAM** references remain mechanism and design evidence. Preserve weighted-vs-macro and ISIC2019-balanced-vs-ordinary distinctions.
- **PanDerm** remains domain-pretraining context and historical evidence, outside the controlled ImageNet comparison; possible downstream/pretraining overlap and cached-Base versus paper-Large differences remain unresolved.
- Repository history supports the need for controls: historical weighted B0+CBAM 84.56% accuracy/.7569 macro-F1; oversampling 85.30%/.7321; focal 83.77%/.7238; weighted/focal fusion 84.36%/.7718; prior-corrected oversampling 85.89%/.7227. Stronger augmentation also regressed in the exploratory work. Accuracy increases alone can conceal balanced-performance losses.
- Saved historical leaders remain **87.69% strict validation accuracy** and **90.75% exploratory validation accuracy**; best strict macro-F1 `.7837` belongs to a different combination. The exploratory split has 563 train/validation shared lesions and is not comparable to the strict benchmark. None of these is a final locked-test result or a new structured experiment. Provenance: `research/legacy_highlights.json`, `results/master_experiment_registry.csv` and historical comparison files.

## Literature structure for the paper

1. Dataset characteristics, imbalance and evaluation leakage risks: HAM10000/base-paper protocol audit.
2. Pretrained CNN comparisons and compute constraints: lightweight/residual/dense/modern families.
3. Attention methods: CBAM mechanism and matched ablation, with soft/patch/class-wise context.
4. Balancing/augmentation/preprocessing: training-only interventions and class-wise trade-offs.
5. Fusion/ensembles: complementarity, constrained selection and cost; retain negative results.
6. XAI/calibration/generalization: representative explanations, uncertainty and dataset limitations.
7. Study gap: our repository lacks a controlled common-recipe backbone comparison and isolated CBAM control. Proposed contribution is reproducible evidence under a fixed lesion-disjoint budget, **not a claim of unprecedented architecture or guaranteed93% accuracy**.

See the resulting [Phase 3 plan and first run proposal](../../phase3/PLAN.md). No new training, predictions or test evaluation occurred during this review.
