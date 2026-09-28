"""Predict one image with the four-stream strict-validation-selected ensemble."""
import argparse
import json

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from src.data import CLASSES, ROOT, transform as b0_transform
from src.models import EfficientNetCBAM
from scripts.probe_panderm_base import TRANSFORM as panderm_transform, load_backbone


RUNS = ("efficientnet_b0_cbam_weighted_v1", "efficientnet_b0_cbam_oversampled_v1")
SVM = ROOT / "checkpoints/accuracy_exploration/panderm_base_strict_heads_v1/standardized_rbf_svc_c10.joblib"
PILOT = ROOT / "checkpoints/accuracy_exploration/panderm_base_last2_pilot_v1/best.pt"


def predict(image, device):
    streams = []
    b0_pixels = b0_transform(size=224)(image).unsqueeze(0).to(device)
    for run in RUNS:
        saved = torch.load(ROOT / "checkpoints" / run / "best.pt", map_location="cpu", weights_only=False)
        model = EfficientNetCBAM(pretrained=False).to(device).eval()
        model.load_state_dict(saved["model"])
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(b0_pixels)
        streams.append(F.softmax(logits.float(), dim=1).cpu().numpy()[0])
        del model, saved
    model = load_backbone(device)
    panderm_pixels = panderm_transform(image).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
        embedding = model.forward_features(panderm_pixels, is_train=False)
    svm = joblib.load(SVM)
    if not np.array_equal(svm[-1].classes_, np.arange(len(CLASSES))):
        raise ValueError("PanDerm SVM class order changed")
    streams.append(svm.predict_proba(embedding.float().cpu().numpy())[0])
    saved = torch.load(PILOT, map_location="cpu", weights_only=False)
    model.head = nn.Linear(768, len(CLASSES)).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
        logits = model(panderm_pixels)
    streams.append(F.softmax(logits.float(), dim=1).cpu().numpy()[0])
    probabilities = np.mean(streams, axis=0)
    return {"predicted_class": CLASSES[int(probabilities.argmax())],
            "probabilities": {name: float(probabilities[index]) for index, name in enumerate(CLASSES)},
            "streams": [*RUNS, "panderm_base_rbf_svc", "panderm_base_last2_pilot"],
            "stream_weights": [.25] * 4,
            "protocol": "selected on strict lesion-disjoint validation; locked test not evaluated"}


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
