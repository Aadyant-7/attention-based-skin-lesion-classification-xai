"""Frozen PanDerm Base linear probe on strict train/validation images only.

The third-party model code and weights stay in the ignored .cache directory.
Fetch them from the authors' PanDerm repository and linked Base checkpoint.
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data import CLASSES, ROOT, SPLIT, image_paths
from src.metrics import classification_metrics
from src.utils import write_json


MODEL_SOURCE = ROOT / ".cache/research_panderm/classification/models/modeling_finetune.py"
WEIGHTS = ROOT / ".cache/panderm_bb_data6_checkpoint-499.pth"
CACHE = ROOT / ".cache/panderm_base_strict_lp_v1"
OUTPUT = ROOT / "results/accuracy_exploration/panderm_base_strict_lp_v1"
HEAD = ROOT / "checkpoints/accuracy_exploration/panderm_base_strict_lp_v1/head.joblib"
TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize((.485, .456, .406), (.228, .224, .225)),
])


class Images(Dataset):
    def __init__(self, rows, paths, transform=TRANSFORM):
        self.rows = rows.reset_index(drop=True)
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.paths[row.image_id]) as image:
            return self.transform(image.convert("RGB")), int(row.label)


def load_backbone(device):
    if not MODEL_SOURCE.is_file() or not WEIGHTS.is_file():
        raise FileNotFoundError("PanDerm source or Base checkpoint missing from .cache; see docs/panderm_probe.md")
    spec = importlib.util.spec_from_file_location("panderm_modeling", MODEL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # timm's model registration looks up its module
    spec.loader.exec_module(module)
    model = module.panderm_base_patch16_224()
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["head.weight", "head.bias"] or unexpected:
        raise ValueError(f"PanDerm weights incompatible: missing={missing}, unexpected={unexpected}")
    model.head = torch.nn.Identity()
    return model.to(device).eval()


def features(model, rows, paths, split, device):
    path = CACHE / f"{split}.npz"
    expected_ids = rows.image_id.to_numpy(dtype=str)
    expected_y = rows.label.to_numpy()
    if path.exists():
        with np.load(path) as saved:
            if np.array_equal(saved["ids"], expected_ids) and np.array_equal(saved["y"], expected_y):
                return saved["features"]
        raise ValueError(f"Stale feature cache: {path}")
    batches = DataLoader(Images(rows, paths), batch_size=32, shuffle=False,
                         num_workers=2, pin_memory=True)
    vectors = []
    with torch.inference_mode():
        for index, (pixels, _) in enumerate(batches):
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                embedding = model.forward_features(pixels.to(device, non_blocking=True), is_train=False)
            vectors.append(embedding.float().cpu().numpy())
            if (index + 1) % 50 == 0:
                print(json.dumps({"split": split, "batches_complete": index + 1,
                                  "batches_total": len(batches)}), flush=True)
    x = np.concatenate(vectors)
    if x.shape != (len(rows), 768) or not np.isfinite(x).all():
        raise ValueError("Invalid extracted PanDerm features")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=expected_ids, y=expected_y, features=x)
    return x


def main():
    started = time.time()
    frame = pd.read_csv(SPLIT)
    train = frame.loc[frame.split == "train"].reset_index(drop=True)
    val = frame.loc[frame.split == "val"].reset_index(drop=True)
    test = frame.loc[frame.split == "test"]
    if len(train) != 7009 or len(val) != 1503 or len(test) != 1503:
        raise ValueError("Strict split sizes changed")
    if set(train.lesion_id) & set(val.lesion_id) or set(train.lesion_id) & set(test.lesion_id) or set(val.lesion_id) & set(test.lesion_id):
        raise ValueError("Strict lesion-disjoint split changed")
    paths, duplicates = image_paths()
    if duplicates or any(image_id not in paths for image_id in pd.concat([train, val]).image_id):
        raise ValueError("Missing or duplicate development images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_backbone(device)
    train_x = features(model, train, paths, "train", device)
    val_x = features(model, val, paths, "val", device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    classifier = LogisticRegression(C=768 * len(CLASSES) / 100, max_iter=1000,
                                    solver="lbfgs", random_state=42)
    classifier.fit(train_x, train.label.to_numpy())
    probabilities = classifier.predict_proba(val_x)
    if not np.array_equal(classifier.classes_, np.arange(len(CLASSES))):
        raise ValueError("Classifier class order changed")
    baselines = {}
    for run in ("efficientnet_b0_cbam_weighted_v1", "efficientnet_b0_cbam_oversampled_v1"):
        with np.load(ROOT / ".cache/phase4" / f"{run}_val_probabilities.npz") as saved:
            if not np.array_equal(saved["ids"], val.image_id.to_numpy(dtype=str)) or not np.array_equal(saved["y"], val.label.to_numpy()):
                raise ValueError("Strict baseline cache order differs")
            baselines[run] = saved["normal"]
    candidates = {
        "panderm_linear": probabilities,
        "panderm_plus_weighted_b0_equal": (probabilities + baselines["efficientnet_b0_cbam_weighted_v1"]) / 2,
        "panderm_plus_oversampled_b0_equal": (probabilities + baselines["efficientnet_b0_cbam_oversampled_v1"]) / 2,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, probs in candidates.items():
        metrics = classification_metrics(val.label.to_numpy(), probs.argmax(axis=1))
        write_json(OUTPUT / f"{name}.json", metrics)
        print(json.dumps({"method": name, "accuracy": metrics["accuracy"],
                          "macro_f1": metrics["macro_f1"]}), flush=True)
    HEAD.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, HEAD)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "val_probabilities.npz", ids=val.image_id.to_numpy(dtype=str),
                        y=val.label.to_numpy(), probabilities=probabilities)
    digest = hashlib.sha256()
    with WEIGHTS.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    write_json(OUTPUT / "protocol.json", {
        "backbone": "PanDerm Base ViT-B/16 frozen", "weights_sha256": digest.hexdigest(),
        "weights_source": "https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB/view",
        "source_code": "https://github.com/SiyuanYan1/PanDerm",
        "input": "Resize short edge to 256, center crop 224, author ImageNet normalization",
        "classifier": "sklearn multinomial logistic regression, C=53.76, max_iter=1000, no class weighting",
        "train_images": len(train), "validation_images": len(val), "test_images_evaluated": 0,
        "train_validation_lesion_overlap": 0, "backbone_updated": False,
        "validation_warning": "Candidates inspected on strict validation; locked test remains untouched",
        "runtime_seconds": time.time() - started,
    })


if __name__ == "__main__":
    main()
