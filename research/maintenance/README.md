# Read-only organization checks

This is the current cleanup verification tool. Historical training utilities remain at their recorded paths; no frozen runtime was changed for organization.

From the project root, run:

```powershell
& '.\.venv\Scripts\python.exe' -B -m research.maintenance.verify_organization --full-hash
```

Checks the pre-cleanup local snapshot, exact archived navigation, approved move/delete ledgers, protected file metadata and SHA-256 hashes, registry references, reporting sources, active Markdown links, Python syntax, CPU imports and R202/R203 signatures. It does not decode images, load model checkpoints, perform inference, start training or initialize CUDA. Hashing large preserved files takes time and reads their bytes only.

The baseline stays local at `.cache/cleanup/20261011/before_manifest.json`; its digest and outcomes are recorded in the [dated cleanup record](../../legacy/organization/cleanup_2026-10-11/README.md). The unchanged environment is checked by metadata against the verified migration baseline; research/data/checkpoint/cache content receives fresh hashes. Run Python with `-B` to avoid recreating bytecode.

`--portable` skips the local cleanup snapshot, but still requires the existing local scientific assets and matching dependencies for registry/import/freeze checks. It is not a checkpoint-free fresh-clone test. The older `research/verify_layout.py` has phase-specific assumptions and writes historical receipts; it is preserved, not used for this cleanup.

## Later authorized work — 11 October 2026

The exact S83 XAI task adds new explanation files and updates active navigation after the dated cleanup. The `--full-hash` baseline authenticates the cleanup's original state and is not expected to accept every later authorized addition/navigation change. Preserve its original receipts. Use `--portable` for current read-only link/reference/import checks, and the new `results/final_s83_xai/v1/verification.json` for exact-method XAI verification. Neither is authorization to regenerate explanations or evaluate the test.
