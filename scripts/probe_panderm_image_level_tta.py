"""Orientation-averaged PanDerm inference on exploratory validation only."""
import csv
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from src.data import ROOT, SPLIT, image_paths
from src.metrics import classification_metrics
from src.utils import write_json
from scripts.probe_panderm_base import TRANSFORM, load_backbone
from scripts.probe_panderm_image_level import BASE, STRONG, SPLIT_PATH, cached_b0


CACHE = ROOT / ".cache/panderm_base_image_level_v1"
OUTPUT = ROOT / "results/exploratory/panderm_base_image_level_tta_v1"
CHECKPOINT = ROOT / "checkpoints/exploratory/panderm_base_image_level_v1"


def main():
    strict = pd.read_csv(SPLIT)
    frame = pd.read_csv(SPLIT_PATH)
    val = frame.loc[frame.split == "val"].reset_index(drop=True)
    if len(val) != 1503 or set(val.image_id) & set(strict.loc[strict.split == "test", "image_id"]):
        raise ValueError("Locked test membership changed")
    paths, duplicates = image_paths()
    if duplicates or any(image_id not in paths for image_id in val.image_id):
        raise ValueError("Missing or duplicate validation images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_backbone(device)
    # Flips are deterministic; no validation label guides individual predictions.
    transforms_by_name = {
        "hflip": transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), TRANSFORM]),
        "vflip": transforms.Compose([transforms.RandomVerticalFlip(p=1.0), TRANSFORM]),
        "hvflip": transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                                       transforms.RandomVerticalFlip(p=1.0), TRANSFORM]),
    }
    ids = val.image_id.to_numpy(dtype=str)
    y = val.label.to_numpy()
    vectors = {}
    with np.load(CACHE / "val.npz") as saved:
        if not np.array_equal(saved["ids"], ids) or not np.array_equal(saved["y"], y):
            raise ValueError("Normal PanDerm cache ordering differs")
        vectors["normal"] = saved["features"]
    for name, transform in transforms_by_name.items():
        # The shared extractor accepts a Dataset transform through its standard
        # image class; load flipped images under a distinct cache key.
        vectors[name] = flipped_features(model, val, paths, name, transform, device)
    del model
    streams = {}
    for name in ("panderm_linear", "panderm_rbf_svc"):
        head = joblib.load(CHECKPOINT / f"{name}.joblib")
        if not np.array_equal(head[-1].classes_, np.arange(7)):
            raise ValueError("PanDerm class order changed")
        streams[name] = {view: head.predict_proba(x) for view, x in vectors.items()}
    b0 = [cached_b0(run, size, ids, y) for run in (BASE, STRONG) for size in (224, 384)]
    b0_sum = sum(b0)
    svc_tta = sum(streams["panderm_rbf_svc"].values()) / 4
    linear_tta = sum(streams["panderm_linear"].values()) / 4
    candidates = {
        "b0_four_control": b0_sum / 4,
        "panderm_svc_tta4": svc_tta,
        "b0_four_plus_svc_tta_five_equal": (b0_sum + svc_tta) / 5,
        "b0_four_plus_svc_tta_plus_linear_tta_six_equal": (b0_sum + svc_tta + linear_tta) / 6,
        "b0_four_plus_four_svc_views_eight_equal": (b0_sum + sum(streams["panderm_rbf_svc"].values())) / 8,
        "b0_four_plus_svc_tta_plus_linear_normal_six_equal":
            (b0_sum + svc_tta + streams["panderm_linear"]["normal"]) / 6,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, probabilities in candidates.items():
        metrics = classification_metrics(y, probabilities.argmax(axis=1))
        write_json(OUTPUT / f"{name}.json", metrics)
        row = {"method": name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
        summaries.append(row)
        print(json.dumps(row), flush=True)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(OUTPUT / "protocol.json", {
        "strict_test_evaluated": False, "model_weights_updated": False,
        "validation_images": len(val), "views": list(vectors),
        "selection_warning": "Six candidates inspected on exploratory image-level validation; shared lesions and validation selection remain",
    })


def flipped_features(model, rows, paths, name, transform, device, cache_dir=CACHE):
    from torch.utils.data import DataLoader
    from scripts.probe_panderm_base import Images

    cache_path = cache_dir / f"val_{name}.npz"
    ids, y = rows.image_id.to_numpy(dtype=str), rows.label.to_numpy()
    if cache_path.exists():
        with np.load(cache_path) as saved:
            if np.array_equal(saved["ids"], ids) and np.array_equal(saved["y"], y):
                return saved["features"]
        raise ValueError(f"Stale flipped feature cache: {cache_path}")
    batches = DataLoader(Images(rows, paths, transform), batch_size=32, shuffle=False,
                         num_workers=2, pin_memory=True)
    vectors = []
    with torch.inference_mode():
        for pixels, _ in batches:
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                embedded = model.forward_features(pixels.to(device, non_blocking=True), is_train=False)
            vectors.append(embedded.float().cpu().numpy())
    x = np.concatenate(vectors)
    if x.shape != (len(rows), 768) or not np.isfinite(x).all():
        raise ValueError("Invalid PanDerm flipped features")
    np.savez_compressed(cache_path, ids=ids, y=y, features=x)
    print(json.dumps({"view": name, "images": len(rows)}), flush=True)
    return x


if __name__ == "__main__":
    main()
