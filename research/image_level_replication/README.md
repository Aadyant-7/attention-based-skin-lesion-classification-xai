# Image-level paper-guided study â€” separate v1 workflow

**Status: Phase 1 completed and independently verified. No model trained and no new accuracy result.**

[Phase 1 closeout and actual overlap counts](PHASE1_CLOSEOUT.md).

This namespace starts a new study using the guide's 70/15/15 proportions, allowing naturally occurring overlap of lesion IDs while keeping original image identities disjoint. Old strict/exploratory results, checkpoints, manifests and registries remain historical evidence. See [the saved plan](PLAN.md).

| Location | Contents |
|---|---|
| `research/image_level_replication/` | Plan, preparation/audit code and frozen protocol JSON. |
| `data/splits/image_level_replication/v1/` | New original-image assignments. |
| `results/image_level_replication/v1/protocol/` | Split audit, class counts, overlap, preservation records and PNG/PDF figures. |
| `results/image_level_replication/v1/experiment_registry.csv` | New study's run registry; header only until a real run exists. |
| `checkpoints/image_level_replication/v1/` | Reserved future local checkpoints; no checkpoint directory created in Phase 1. |

## Metadata-only commands

```powershell
Set-Location 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
.\.venv\Scripts\python.exe -B -m research.image_level_replication.prepare_protocol --prepare
.\.venv\Scripts\python.exe -B -m research.image_level_replication.prepare_protocol --verify
```

Preparation freezes one seed-42 split and is idempotent. Verification is read-only. Neither command opens images, imports torch, uses the GPU, trains a model or evaluates test performance. Missing/changed artifacts fail rather than silently recreating assignments.

Future training can use `load_development()` to obtain only train/validation rows after frozen-manifest checks. This is a metadata helper, not an implemented training or test-inference runner. Phase 2 must prepare that isolated runner/config before launch approval.

**Fresh external pretrained initialization is required.** Legacy project checkpoints and probabilities contain images reassigned to this study's test and cannot serve as its components. The study is post-development and image-level, not a pristine unseen-lesion test. Original S83 results remain 94.011976% validation / 87.558217% test audit.
