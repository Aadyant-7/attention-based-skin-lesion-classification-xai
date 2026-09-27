# Phase 4 validation error comparison

Best new candidate by macro F1: `weighted+focal|weighted_tta|0.9,0.1` (accuracy 0.843646; macro F1 0.771810).

Counts below combine both directions of each confusion pair on the 1,503-image validation partition.

| Pair | Weighted normal | Weighted flip TTA | Oversampled normal | Best new candidate |
|---|---:|---:|---:|---:|
| mel ↔ nv | 98 | 103 | 79 | 104 |
| bkl ↔ nv | 31 | 36 | 38 | 34 |
| bkl ↔ mel | 33 | 33 | 30 | 32 |
| akiec ↔ bcc | 18 | 12 | 13 | 12 |

Against weighted flip TTA, the best new candidate reduces bkl↔nv from 36 to 34 and bkl↔mel from 33 to 32; akiec↔bcc stays at 12, while mel↔nv worsens from 103 to 104. Against weighted normal, akiec↔bcc improves from 18 to 12, but mel↔nv worsens from 98 to 104. Against oversampled normal, bkl↔nv improves from 38 to 34 and akiec↔bcc from 13 to 12, while mel↔nv and bkl↔mel worsen.

Per-class F1 for the best new candidate: akiec 0.660, bcc 0.780, bkl 0.751, df 0.914, mel 0.547, nv 0.921, vasc 0.829.

All selection and error analysis used validation labels only; no test features or predictions were produced.
