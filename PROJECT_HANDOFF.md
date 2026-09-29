# Minor Project handoff (repository snapshot: 29 September 2026)

**Project:** Attention-Based Deep Learning Framework for Skin Lesion Classification with Explainable AI. This is a seven-class, image-only HAM10000 research project. The agreed final methodology is **EfficientNet variants + CBAM + Grad-CAM**. PanDerm was an explicitly separate research probe, not part of that final architecture. The original locked test set has **never been evaluated**. Every accuracy below is a **validation** result, and no 93%+ strict or final-test result has been established.

## Repository and evaluation rules

- Local root: `C:\Users\MARS\Desktop\College\Minor Project\Project Development` (paths below are relative to this root; a friend's clone will have a different absolute root).
- GitHub: `https://github.com/Aadyant-7/attention-based-skin-lesion-classification-xai.git`; current branch: **`accuracy-exploration`**. `main` retains the earlier strict benchmark snapshot. For current results, clone/select `accuracy-exploration`.
- HAM10000 has 10,015 dermoscopic images, seven labels (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`) and 7,470 lesion IDs. Class imbalance is severe (6,705 `nv` versus 115 `df` original images).
- **Strict protocol:** fixed, lesion-ID-disjoint 7,009 train / 1,503 validation / 1,503 locked-test images. Related images of one lesion stay in one partition. Training-only weights or oversampling are computed after splitting. The split is in `data/splits/split_assignments.csv`; `data/splits/label_mapping.json` holds labels.
- **Exploratory image-level protocol:** re-splits only the original 8,512-image train+validation development pool into 7,009 train / 1,503 validation, leaving all original test images untouched. It has **563 lesion IDs** crossing train/validation, affecting **596 validation images**. Its scores are weaker-protocol, validation-selected results; never present them as lesion-independent or test accuracy. Manifest: `data/splits/exploratory/image_level_dev_v1.csv`.
- The final model/configuration is not locked. Its explanation review and one-time untouched-test evaluation remain pending. Do not run a test evaluation or new training merely to answer a documentation question.

## Research and what was learned

- The HAM10000 dataset paper established lesion identities and motivates grouped splitting. The base paper by Pardede et al. reported 98.86% EfficientNet-B0 accuracy with balancing and Grad-CAM, but its published test table totals **7,041 balanced cases**, unlike about 1,502 untouched original HAM10000 images in a 15% test set. Its exact processing cannot be reconstructed from the paper; the percentage is **not directly comparable** to this project.
- Haque et al. reported 91.15% accuracy / 0.8545 macro-F1 with a much larger 384-pixel EfficientNetV2-L attention approach; lesion grouping is not established. A public EfficientNet-B2 implementation uses an image-level split and stronger augmentation. Other 93–96% claims often use different class counts, cohorts, historical splits, or retrospective evaluation. See `docs/accuracy_exploration_research.md`, `docs/high_accuracy_claims_audit.md`, and `docs/base_paper_protocol_audit.md` for source-specific details.
- The implemented core classifier fine-tunes ImageNet-pretrained EfficientNet-B0 on 224×224 images, puts one CBAM block after its final spatial feature map, then global-average-pools and predicts seven classes. Initial training used AdamW, batch size 64, up to 20 epochs, and macro-F1-selected `best.pt`. Grad-CAM is a separate post-hoc explanation of a prediction; CBAM changes features during inference. Code: `src/models.py`, `src/cbam.py`, `src/train.py`, `src/gradcam.py`; configs: `configs/first_run.json`, `configs/oversampled_v1.json`, `configs/focal_v1.json`.

## Completed experiments and saved validation results

| Protocol and method | Accuracy | Macro-F1 | Main conclusion |
| --- | ---: | ---: | --- |
| Strict weighted-loss B0 + CBAM | **84.56%** | **0.7569** | Original trained core baseline; completed 20 epochs, best epoch 16. |
| Strict full oversampling B0 + CBAM | 85.30% | 0.7321 | Accuracy rose but class-balanced performance fell; repeated-reference overfitting. |
| Strict focal-loss B0 + CBAM | 83.77% | 0.7238 | Did not beat weighted baseline. |
| Strict weighted + focal probability ensemble | 84.36% | 0.7718 | Improved macro-F1 without training a new CNN. |
| Strict prior-corrected oversampled model | 85.89% | 0.7227 | Higher accuracy came with poorer macro-F1; not a balanced solution. |
| Exploratory image-level weighted B0 + CBAM | 86.23% | 0.7808 | Protocol probe; shared-lesion validation. |
| Exploratory same-split stronger augmentation | 85.30% | 0.7657 | Reduced memorization but worsened both main metrics. |
| Exploratory four B0 + CBAM resolution streams | **88.16%** | **0.8280** | Highest listed EfficientNet + CBAM exploratory ensemble; reused two saved checkpoints at 224/384 pixels. |
| Strict PanDerm/B0 ensemble, accuracy leader | **87.69%** | 0.7771 | Highest lesion-disjoint validation accuracy; PanDerm is outside final architecture. |
| Strict different PanDerm/B0 combination, macro-F1 leader | 87.03% | **0.7837** | Highest strict macro-F1; **not** the 87.69% configuration. |
| Exploratory eight B0/PanDerm streams | **90.75%** | **0.8573** | Highest numeric validation score, but shared-lesion image-level split and outside final architecture. **Not a test result.** |

Other completed, inexpensive analyses: horizontal-flip test-time averaging, saved-checkpoint probability ensembles, frozen post-CBAM feature classifiers, image/metadata fusion feasibility, MC-dropout, crop/preprocessing checks, and post-hoc prior correction. Metadata did not meet its improvement threshold, and these checks did not establish a broadly better strict EfficientNet + CBAM model. A two-epoch PanDerm partial fine-tune pilot stopped at its preset threshold; PanDerm's frozen features complemented B0 in ensembles. Saved checkpoints/results were retained. For the full journey and interpretations, see `reports/Project_Draft_Report_Revised.*` and `reports/Team_Layman_Project_Guide_Revised.*`.

## Exact file map (and what a GitHub clone contains)

| Path | Purpose / availability |
| --- | --- |
| `README.md` | Project setup, architecture, and data policy; tracked. |
| `docs/accuracy_leaderboard.md` | Best strict versus exploratory results; tracked. `docs/panderm_probe.md` and `docs/panderm_image_level_followup.md` explain PanDerm methods. |
| `results/experiments.csv` | Three original trained B0 + CBAM run configurations, accuracies, F1, checkpoint paths, status; tracked. |
| `results/validation_inference_comparison.csv`, `results/phase4/`, `results/phase5_multimodal_probe/`, `results/phase5b_inference_audit/`, `results/phase5c_prior_correction/` | Inference, frozen-feature, metadata, dropout, and correction comparisons; tracked CSV/JSON/Markdown. |
| `results/exploratory/multires_ensemble/summary.csv` and `four_equal.json` | 88.16% four-stream B0 result; JSON has per-class scores and `confusion_matrix`; tracked. |
| `results/exploratory/panderm_base_image_level_tta_v1/summary.csv` and `b0_four_plus_four_svc_views_eight_equal.json` | 90.75% exploratory result, per-class scores and confusion matrix; tracked. |
| `results/accuracy_exploration/panderm_base_strict_tta_v1/summary.csv`, `prior_four_with_svc_tta_equal.json`, `weighted_b0_plus_svc_tta_two_equal.json` | 87.69% strict accuracy leader and separate 0.7837 macro-F1 leader, with confusion matrices; tracked. |
| `results/runs/<run_name>/history.csv`, `validation_metrics.json`, logs/config | Detailed original-run records; **local only, Git-ignored**. Example run name: `efficientnet_b0_cbam_weighted_v1`. Some exploratory history/summary files are separately tracked under `results/exploratory/`. |
| `checkpoints/<run_name>/best.pt` and `latest.pt`; `checkpoints/exploratory/`; PanDerm `.joblib` heads | Actual saved model weights/heads; **local only, Git-ignored**. PanDerm source weights are also cached locally, not in GitHub. |
| `data/splits/` | Saved strict/exploratory split manifests and label map; tracked. `data/raw/HAM10000/` contains raw images/metadata locally and is Git-ignored. |
| `src/`, `configs/`, `scripts/` | Model/training/evaluation/XAI implementation and experiment scripts; tracked. `src/evaluate.py` is the explicit locked-test evaluator; do not invoke it before final configuration lock. |
| `reports/` | Six current DOCX/PDF files: revised draft, revised layman guide, updated progress content; committed to GitHub, including explicitly added PDFs. |
| `Presentation_Handoff_Guide.md` and `PROJECT_HANDOFF.md` | Laptop presentation instructions and this compact theory handoff; committed to GitHub. |
| `gradcam_outputs/` | Currently empty; no ready Grad-CAM overlay or presentation plot is saved. Grad-CAM code exists, but systematic visual review remains. |

**For another ChatGPT:** Treat the tracked JSON/CSV and local run records as evidence, keep protocol labels beside every number, and never invent a held-out test score or claim that a checkpoint/report is included in a GitHub clone. A clone of `accuracy-exploration` shows tracked source, manifests, research notes, tables and metric JSONs; copy ignored/local artifacts separately if needed. No raw HAM10000 images or model weights are needed merely to read or present saved results.
