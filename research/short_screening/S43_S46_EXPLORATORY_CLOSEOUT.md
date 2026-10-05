# S43–S46 bounded CPU follow-up

All evaluations reuse the SAME1503-image exploratory validation cohort. No new model training, GPU inference, raw image access or test labels. All outcomes retained, not just the winner.

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| S42 accuracy-selected equal four baseline |93.1470%|.879744|
| strong_convnext_weights |93.0140%|0.879805|
| tiny_emphasis_weights |92.9474%|0.879796|
| equal_geometric_fusion |92.8144%|0.875328|
| S46 fixed ALL macro-F1 checkpoint rule |**93.4797%**|**.883680**|

## Why this is an actual evaluated result without a new GPU run

The original CNN runs already executed inference on every validation image and saved seven-class probability vectors. CPU fusion sums those same vectors and divides by four, then takes the highest-probability class. Accuracy and macro-F1 are calculated against the saved validation labels. This is evaluation of cached model outputs, not an estimate or invented forecast. It is not new training.

The winning S46 uses ConvNeXt-Tiny, ConvNeXt-Small, DenseNet201 and EfficientNetV2-S, all using their previously saved macro-F1-selected checkpoint, weights25% each. Exact checkpoints/prediction files and hashes are in `results/short_screening/s46_all_f1_checkpoint_fusion/candidate_manifest.json`. In this set, Tiny and Dense change compared with their accuracy winners; Small/V2S have the same saved metric winners. No hand-selected epoch mixtures, additional checkpoint combinations or weight search.

S46:1405correct/98incorrect. Compared with S42:10gained/5lost, net5, +.3327pp. Melanoma recall stays79.04%; inspect the full per-class table for all trade-offs. F1 increases from .879744 to .883680.

The predeclared material gate(+.5pp vs S42 with no F1/MEL decline) is not passed. Report a modest developmental improvement, not statistical superiority, an independent test result or a guaranteed95% model. The validation cohort was already used for model/checkpoint selection; choosing among these follow-ups adds further selection optimism. Do not erase this limitation.

## Decision

Preserve S46 as the highest observed exploratory accuracy/F1 among these candidates. Stop this finite sequence; reject the two weighted/geometric alternatives for accuracy. Do not launch another GPU run merely to recompute already saved outputs. All raw/normalized matrices, class scores, predictions/probabilities, PNG/PDF and registry entries are preserved. Original strict frozen method and first/only86.7598%test remain unchanged.
