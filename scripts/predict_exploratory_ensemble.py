"""Predict one image with the frozen exploratory four-stream ensemble.

This model was selected on an image-level validation split and is not a
lesion-disjoint test result. Only load checkpoints from this repository.
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from src.data import CLASSES, ROOT, transform
from src.models import EfficientNetCBAM


RUNS = ("image_level_weighted_b0_cbam_v1", "image_level_strong_aug_b0_cbam_v1")
SIZES = (224, 384)


def predict(image, device):
    streams = []
    for run in RUNS:
        checkpoint = ROOT / "checkpoints/exploratory" / run / "best.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = EfficientNetCBAM(pretrained=False).to(device).eval()
        model.load_state_dict(saved["model"])
        for size in SIZES:
            pixels = transform(size=size)(image).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(pixels)
            streams.append(F.softmax(logits.float(), dim=1).cpu().numpy()[0])
        del model
    probabilities = np.mean(streams, axis=0)
    return {"predicted_class": CLASSES[int(probabilities.argmax())],
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
            "checkpoints": [f"checkpoints/exploratory/{run}/best.pt" for run in RUNS],
            "resolutions": list(SIZES), "stream_weights": [0.25] * 4,
            "protocol": "exploratory image-level validation; not lesion-disjoint test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to a skin-lesion image")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    print(json.dumps({"image": args.image, **predict(image, device)}, indent=2))


if __name__ == "__main__":
    main()
