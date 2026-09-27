"""Explicit final evaluation for a later phase. Never called by train.py."""
import argparse
import json
import torch
from torch import nn

from .data import ROOT, LesionDataset, create_or_load_split
from .metrics import classification_metrics
from .models import EfficientNetCBAM
from .train import loader
from .utils import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-locked-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_locked_test:
        parser.error("Final test evaluation requires --confirm-locked-test after configuration lock")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    rows, paths = create_or_load_split()
    test_rows = rows.loc[rows.split == "test"]
    batches = loader(LesionDataset(test_rows, paths, size=saved["config"]["image_size"]), saved["config"]["batch_size"], 0)
    model = EfficientNetCBAM(pretrained=False).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    truth, predicted = [], []
    with torch.inference_mode():
        for x, y in batches:
            predicted.extend(model(x.to(device)).argmax(1).cpu().tolist())
            truth.extend(y.tolist())
    result = classification_metrics(truth, predicted)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
