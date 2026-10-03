# Phase 1 audit and handoff — 3 October 2026

**Scope completed:** repository audit, legacy preservation/indexing, structured workspace, common model interface and logging contract, dataset/protocol tables, saved-result figures, literature schema and next-phase plan. No new model training, inference experiment, test-image loading or test evaluation was launched. Final architecture is now open to experimental evidence.

## Findings

- The repository contains a functioning historical B0+CBAM training/data/metrics stack, six CNN fine-tuning runs (three original B0 losses, two exploratory B0 recipes, one two-epoch PanDerm pilot), many shallow-head/inference/ensemble candidates, fixed split manifests, local checkpoints, research audits and six current report files.
- The audit indexed **247 historical files (~1.58 GB)**, including local checkpoints and code. `results/legacy/artifact_manifest.csv` records exact paths, hashes, sizes and tracking state. All available historical hashes were checked again after the build.
- Seven groups of identical file contents were found (including duplicate exported histories, repeated best-result JSONs and empty error logs); **none were deleted**. No broken relative Markdown links were found in historical top-level research notes. The raw image filename inventory has 10,015 unique IDs, no missing IDs and no extra IDs. Raw pixels were not read.
- Hardware was queried: RTX 4060, 8 GB VRAM, CUDA available. Installed PyTorch 2.11.0+cu128/torchvision 0.26.0+cu128 support the proposed CNNs. Package versions and cached source/weight inventory are in `results/audit/`.
- Historical naming mixes run summaries, best-of-sweep JSONs, protocol metadata and candidate grids. Registry rows are evidence records, **not a count of independent trained models**. Loss curves across different losses are not directly comparable scalar objectives.
- Old model/training/Grad-CAM code is B0+CBAM-specific. A no-CBAM control was never trained; CBAM's isolated contribution is therefore unproven. Grad-CAM code exists but there are no saved, systematically reviewed examples.
- Legacy training enables cuDNN benchmarking, so a fixed seed alone does not promise full determinism. Its completed-run resume path can skip the epoch loop and reach an undefined `epoch`; do not rerun finished historical training as a presentation step. Both concerns are documented for the new Phase-3 runner rather than silently altering preserved historical code.

## Organization and preservation

No historical experiment, checkpoint, split, script, report or result was moved. A manifest/reference approach avoids breaking embedded paths and copying large model weights. Prior `main` and `accuracy-exploration` Git history are preserved; active infrastructure lives on **`structured-research`**, derived from the complete accuracy-exploration state.

New areas:

```text
research/                         common IO/registry/plots/model/run-artifact APIs
  configs/                        proposed new-run configuration
  literature/                     review schema and evidence index
  tests/                          infrastructure checks only
results/
  legacy/                         immutable inventory + portable small evidence copies
  structured_experiments/         future run packages (no trained structured runs yet)
  datasets/                       strict/exploratory class tables and manifest audit
  figures/legacy/                 historical training/CM/class-score evidence by protocol
  model_comparison/legacy/        separate strict and exploratory historical comparisons
  ablations/                      future controlled attention comparison outputs
  ensembles/                      future controlled ensemble outputs
  final/                          reserved final frozen assessment; no test scores
  audit/                          verification and environment reports
  master_experiment_registry.csv  authoritative evidence index
checkpoints/structured/            future local model weights
NEXT_STEPS.md                      Phase 2–10 plan and model shortlist
```

Small ignored run config/history/metric files were copied, byte-verified, to `results/legacy/evidence/` for a portable evidence pack. Their originals remain intact. Checkpoints, raw HAM10000 images, external source/weights, feature caches, private reference PDFs and environment files remain local/Git-ignored.

## Registry and dataset evidence

The registry has **473 historical evidence rows**, **zero structured trained runs**, and **89 complete metric records** whose accuracy, macro precision/recall/F1, per-class metrics and confusion-matrix counts were checked for mutual consistency. Historical source SHA-256 and split-manifest hashes accompany records. Grid candidates with only aggregate scores retain their CSV row selector; absent metadata stays empty. Four historical leaders have script-verified ensemble members, weights and local checkpoint pointers.

The weighted-run total runtime is missing from its repaired ledger; the PanDerm pilot has epoch elapsed values but no verified whole-run runtime. Both remain unknown. Portable mode preserves registry entries when an ignored original is absent and uses verified small copies; it does not erase missing-source history.

Verified primary class table (also CSV, Markdown, LaTeX fragment, PNG/PDF distribution chart):

| Class | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| nv | 6,705 | 4,694 | 1,006 | 1,005 |
| mel | 1,113 | 779 | 166 | 168 |
| bkl | 1,099 | 769 | 165 | 165 |
| bcc | 514 | 359 | 77 | 78 |
| akiec | 327 | 228 | 50 | 49 |
| vasc | 142 | 100 | 21 | 21 |
| df | 115 | 80 | 18 | 17 |
| **Total** | **10,015** | **7,009** | **1,503** | **1,503** |

All three primary lesion-ID intersections are **zero**. The exploratory manifest has 563 train/validation shared lesion IDs and 596 affected validation images; its original test assignments are identical to the strict manifest. Saved manifests match original metadata labels/lesion IDs. This is a metadata/count audit, not model assessment of test images.

## Visual and literature infrastructure

Ten highlighted historical results now have raw and normalized confusion matrices, per-class score/support charts, metric CSVs and provenance. All available curves are exported; missing pilot training-accuracy/validation-loss values are omitted. Separate historical comparison tables/figures are labelled by protocol and do not pretend to be controlled architecture comparisons. Together with dataset/comparison charts there are **62 figure pairs** (300-dpi PNG plus vector PDF). Start at `results/figures/figure_index.csv`.

The literature CSV template contains the guide's year/authors/dataset/classes/split/model/techniques/metric/XAI/limitations/evidence fields. A historical evidence index links already inspected sources. **A new complete literature review has not been claimed**; verification and writing belong to Phase 2.

## New code boundaries and verification

The model factory supports B0, B2, MobileNetV3-Large, ResNet50, DenseNet121 and ConvNeXt-Tiny with a common seven-class head and optional CBAM. It preserves architecture-native normalization and refuses a changing `DEFAULT` weight alias. The artifact writer stores config/environment/history/metrics, validates class support and selection, writes figures and atomically updates the registry. It rejects old checkpoints as new-run checkpoints and refuses existing run IDs.

**11 infrastructure tests passed**, including concurrent registry writers, incomplete/malformed CSV schemas, portable-source retention, metrics/matrix consistency, same-size wrong-validation-cohort rejection and output shape for all six backbones with/without CBAM. Model checks used random weights and synthetic CPU tensors only; no pretrained weight downloads or model-result measurements occurred. Saved-result figures were visually inspected for layout; historical hashes remained unchanged.

The common training runner/resume implementation, controlled CBAM training, new ensemble evaluation and final XAI/test adaptation are intentionally **later-phase work**. The Phase-1 configuration example is not a launch command.

## Files and next action

**Created:** `research/` infrastructure/docs/tests/config/literature files, `NEXT_STEPS.md`, master registry, legacy manifest/snapshots/portable evidence, dataset tables/audits, figures/comparison tables and audit reports. `.gitattributes` preserves exact bytes for hashed legacy copies and their immutable manifest across Git checkouts. **Modified:** root `README.md` navigation, short current-direction banners in the two handoff guides, and `.gitignore` exceptions for generated result figure PDFs. The three original root documents were copied unchanged into `results/legacy/documentation/` before adding banners/navigation. **Moved:** none. **Untouched:** all historical scripts/source/configs/metrics/checkpoints/reports/raw data/split manifests.

Recommended initial comparison: **B0, MobileNetV3-Large, ResNet50, DenseNet121, ConvNeXt-Tiny**; B2 is an optional sixth after measured timing/memory checks. Installed weight enums/parameter metadata and official source links are in `research/backbone_shortlist.csv` and `NEXT_STEPS.md`. No model's success or full training-memory fit is assumed.

**Next phase:** Phase 2, verify literature rows and freeze the primary comparison recipe. Enough local inputs exist; no user upload or decision is required to complete Phase 1. A target publication template and actual meeting/GPU-availability schedule would refine later formatting/budget, but neither blocks Phase 2. No manual command is required now; rebuild/test commands are in `NEXT_STEPS.md`.
