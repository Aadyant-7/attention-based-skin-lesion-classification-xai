"""Predict with the eight-stream exploratory PanDerm/B0 orientation ensemble."""
import argparse
import json

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from src.data import CLASSES, ROOT, transform as b0_transform
from src.models import EfficientNetCBAM
from scripts.probe_panderm_base import TRANSFORM as panderm_transform, load_backbone


RUNS = ("image_level_weighted_b0_cbam_v1", "image_level_strong_aug_b0_cbam_v1")
HEAD = ROOT / "checkpoints/exploratory/panderm_base_image_level_v1/panderm_rbf_svc.joblib"


def predict(image, device):
    streams = []
    for run in RUNS:
        saved = torch.load(ROOT / "checkpoints/exploratory" / run / "best.pt",
                           map_location="cpu", weights_only=False)
        model = EfficientNetCBAM(pretrained=False).to(device).eval()
        model.load_state_dict(saved["model"])
        for size in (224, 384):
            pixels = b0_transform(size=size)(image).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(pixels)
            streams.append(F.softmax(logits.float(), dim=1).cpu().numpy()[0])
        del model, saved
    model = load_backbone(device)
    head = joblib.load(HEAD)
    if not np.array_equal(head[-1].classes_, np.arange(len(CLASSES))):
        raise ValueError("PanDerm head class order changed")
    views = (image, ImageOps.mirror(image), ImageOps.flip(image),
             ImageOps.flip(ImageOps.mirror(image)))
    for view in views:
        pixels = panderm_transform(view).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            embedding = model.forward_features(pixels, is_train=False)
        streams.append(head.predict_proba(embedding.float().cpu().numpy())[0])
    probabilities = np.mean(streams, axis=0)
    return {"predicted_class": CLASSES[int(probabilities.argmax())],
            "probabilities": {name: float(probabilities[index]) for index, name in enumerate(CLASSES)},
            "streams": [f"{run}_{size}" for run in RUNS for size in (224, 384)]
                       + ["panderm_svc_normal", "panderm_svc_hflip", "panderm_svc_vflip", "panderm_svc_hvflip"],
            "stream_weights": [.125] * 8,
            "protocol": "exploratory image-level validation; shared lesions; locked test not evaluated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"image": args.image, **predict(image, device)}, indent=2))


if __name__ == "__main__":
    main()
