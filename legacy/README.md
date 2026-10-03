# Historical research index

Previous work is preserved and indexed separately from active `research/` experiments.

| Original location | Preserved contents |
|---|---|
| `src/`, `scripts/`, `configs/` | Historical B0+CBAM pipeline, probes, predictors and recipes; unchanged. |
| `results/runs/`, `results/validation_inference/`, `results/phase4/`, `results/phase5*/` | Original training, features, metadata and inference studies. |
| `results/accuracy_exploration/`, `results/exploratory/`, `results/strict_multires_probe/` | PanDerm, image-level and strict multiresolution evidence. |
| `checkpoints/` except `structured/` | Local historical best/latest weights; never moved or overwritten. |
| `docs/`, `legacy/reports/` | Historical notes (original paths) and six archived DOCX/PDF files. |
| `legacy/notebooks/`, `legacy/archive/`, `docs/private_reference/` | Archived notebooks/materials and local references. |
| `.cache/`, `legacy/gradcam_outputs/` | Local features/source/weights and archived old XAI location. |

[Immutable inventory](../results/legacy/artifact_manifest.csv) records paths/sizes/hashes/tracking. [Portable evidence](../results/legacy/portable_evidence_index.csv) indexes exact small config/history/metric copies. [Original documentation snapshots](../results/legacy/documentation/) retain pre-Phase-1 navigation.

The two root handoff guides moved byte-for-byte into `navigation/`; the Phase-1 README was copied before replacement. [Relocation record](navigation/relocation_index.csv) records hashes and old/new paths. Historical paths inside these documents refer to the **project root**; use the current folder map to navigate. The later folder cleanup moved reports/notebooks using `path_map.json`; original manifest rows/hashes remain unchanged. See [cleanup report](organization/README.md).
