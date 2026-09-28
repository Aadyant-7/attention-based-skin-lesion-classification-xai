# Accuracy leaderboard by evaluation protocol

Updated 2026-09-28. All figures below are **validation** results. The project's locked test partition remains untouched. The displayed best across all methods is the largest numeric accuracy, with its protocol stated; scores across protocols are not directly comparable.

| Protocol | Method | Accuracy | Macro F1 | Checkpoint status |
|---|---|---:|---:|---|
| Exploratory image-level (shared lesions) | **Four B0 resolution streams + four PanDerm SVM orientation views, equal average** | **0.907518** | **0.857301** | Two B0 `best.pt` files and PanDerm SVM head saved locally; PanDerm source weights cached. |
| Exploratory image-level (shared lesions) | Four B0 streams + two PanDerm normal-view heads, equal average | 0.898204 | 0.836769 | Components saved locally. |
| Exploratory image-level (shared lesions) | Four B0 resolution streams, equal average | 0.881570 | 0.828008 | Both B0 `best.pt` files saved locally. |
| Strict lesion-disjoint | **Prior four-stream PanDerm/B0 ensemble with PanDerm SVM orientation average** | **0.876913** | 0.777119 | B0 and PanDerm pilot `best.pt`, PanDerm SVM head, and source weights saved locally. |
| Strict lesion-disjoint | Prior PanDerm pilot + SVM + two B0, equal average | 0.876248 | 0.778083 | Components saved locally. |
| Strict lesion-disjoint | Weighted B0 + PanDerm SVM orientation average | 0.870259 | **0.783666** | Components saved locally; highest strict macro F1. |

The highest **numeric** accuracy so far is **90.75% exploratory validation**. The highest **strict** accuracy is **87.69% validation**. The exploratory split shares 563 lesion IDs between training and validation, so its 90.75% must be labeled as such in the research paper. Full per-class metrics and histories are linked from [the PanDerm follow-up](panderm_image_level_followup.md) and [the strict PanDerm probe](panderm_probe.md). The [base-paper protocol audit](base_paper_protocol_audit.md) explains why its reported 98.86% is not a like-for-like target.
