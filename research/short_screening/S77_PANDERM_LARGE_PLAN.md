# S77: published PanDerm Large, matched against Base

Frozen before scoring. Question: does the published dermatology-pretrained Large encoder supply substantially better features than the smaller Base used previously?

- Official [PanDerm source](https://github.com/SiyuanYan1/PanDerm) and [paper](https://www.nature.com/articles/s41591-025-03747-y). Large: ViT-L/16, 303,326,208 encoder parameters, 1,024-dimensional CLS features. Base: 85,807,872 parameters, 768 dimensions.
- Both use FP32 frozen inference, author resize-256/center-crop-224 and normalization, the same 7,009 training / 1,503 exploratory validation images, and the same training-only StandardScaler plus RBF SVM (C=10, gamma=scale, seed=42). No backbone training or hyperparameter search.
- The older Base evidence included both a frozen SVM and a two-epoch strict pilot updating the last two blocks. It was incorrect to describe all previous PanDerm work as frozen-only. Neither was a completed exploratory Large fine-tuning experiment.
- CPU preflight verified strict encoder loads, finite features, exact kernel equivalence and zero original-test image/lesion overlap. Only Large's documented non-encoder pretraining heads are excluded.
- Conditional S78: only if Large beats matched Base by at least 1 percentage point, reaches 90%, and does not lower macro-F1/melanoma recall, replace B0 in S53 once. Keep the other four members unchanged and all five weights at 0.2. Advance only for at least +0.5 percentage points / eight net correct predictions with nondecreasing macro-F1/melanoma recall. No rescue weight search.
- Budget: at most 10 minutes feature extraction and 15 minutes total execution. Complete caches/checkpoints are preserved on interruption. No original-test labels/images are used.

Large checkpoint SHA256: `23405e92f853c6f7ac9da80ccf4cfde84d4feb22cd9d7b98c7e850e3875554b2`. Full Base/Large/source/split hashes and official download links are in the saved signature and plan. Third-party weights/source stay local and retain their original licensing; they are not redistributed in Git.

Run from project root:
```powershell
.\.venv\Scripts\python.exe -B -u -m research.short_screening.panderm_large_screen --run
Get-Content .\results\short_screening\s77_panderm_large_transfer\run.log -Tail 20 -Wait
```

Results: `results/short_screening/s77_panderm_large_transfer/`. Local fitted heads: `checkpoints/short_screening/s77_panderm_large_transfer/`. Frozen feature caches: `.cache/s77_panderm_large_transfer/`.

Limitations: this is fixed frozen transfer, not a replication of the paper's fine-tuning protocol. Complete pretraining overlap with HAM10000 cannot independently be ruled out. Validation has already been reused extensively. No score here establishes independent test accuracy.
