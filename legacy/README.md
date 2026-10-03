# Historical research index

Previous work is preserved and indexed separately from active `research/` experiments.

| Original location | Preserved contents |
|---|---|
| `src/`, `scripts/`, `configs/` | Historical B0+CBAM pipeline, probes, predictors and recipes; unchanged. |
| `results/runs/`, `results/validation_inference/`, `results/phase4/`, `results/phase5*/` | Original training, features, metadata and inference studies. |
| `results/accuracy_exploration/`, `results/exploratory/`, `results/strict_multires_probe/` | PanDerm, image-level and strict multiresolution evidence. |
| `checkpoints/` except `structured/` | Local historical best/latest weights; never moved or overwritten. |
| `docs/`, `reports/` | Historical research notes and six existing DOCX/PDF report files. |
| `notebooks/legacy/`, `archive/`, `docs/private_reference/` | Local notebooks, archive and private references. |
| `.cache/`, `gradcam_outputs/` | Local features/source/weights and old XAI output location. |

[Immutable inventory](../results/legacy/artifact_manifest.csv) records paths/sizes/hashes/tracking. [Portable evidence](../results/legacy/portable_evidence_index.csv) indexes exact small config/history/metric copies. [Original documentation snapshots](../results/legacy/documentation/) retain pre-Phase-1 navigation.

The two root handoff guides moved byte-for-byte into `navigation/`; the Phase-1 README was copied before replacement. [Relocation record](navigation/relocation_index.csv) records hashes and old/new paths. Historical paths inside these documents refer to the **project root**; use the current folder map to navigate. No immutable experiment asset was relocated.
