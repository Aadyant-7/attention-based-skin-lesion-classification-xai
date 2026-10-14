# Read-only repository and preservation checks

## Any fresh clone (no ML dependencies or local assets)

From the clone root:

```powershell
python -B -m research.maintenance.verify_repository
```

Uses only the Python standard library. Checks Python syntax, current navigation, registry references with explicit local-only exclusions, reporting source SHA-256 values, frozen S83 receipts and saved XAI gallery/resources. It does not import Torch, decode images, load weights, run inference, score the test or write files by default. `--output <path>` saves a receipt only when explicitly requested.

## Existing full project (original environment and scientific assets)

```powershell
& '.\.venv\Scripts\python.exe' -B -m research.maintenance.verify_organization --portable
```

The historical option name `--portable` skips the old local cleanup snapshot. This check still requires local checkpoint/cache/raw metadata references and the matching ML environment; it is **not** the fresh-clone check above. It verifies registry references, reporting hashes, syntax, reviewed CPU imports and R202/R203 frozen signatures, with CUDA uninitialized. It loads no model or image.

## Historical cleanup baseline

The 11 October baseline and `--full-hash` option authenticate that dated organization snapshot, not the latest repository. Later authorized XAI, research and navigation additions can differ. Preserve the original [cleanup record](../../legacy/organization/cleanup_2026-10-11/README.md) and local baseline under `.cache/cleanup/20261011/`; do not rewrite them to make a modern check pass.

`research/verify_layout.py` also has phase-specific historical assumptions and writes old audit receipts; it is not a current gate. Do not run training/sanity or XAI regeneration commands as browsing checks. The XAI runner's `--verify` rewrites receipts; the new clone audit checks existing hashes without invoking it.

See [setup and excluded assets](../../docs/SETUP.md), [current status](../../docs/project/STATUS.md) and [final audit](../../docs/project/FINAL_AUDIT.md). No frozen runtime, checkpoint or metric is changed by these navigation tools.
