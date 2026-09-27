"""Grad-CAM visualization for the post-training model (explanation only)."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize

from .data import MEAN, STD, transform
from .models import EfficientNetCBAM


def gradcam(model, image_tensor, class_index=None):
    activations, gradients = [], []
    target = model.features[-1]
    forward = target.register_forward_hook(lambda module, inputs, output: activations.append(output))
    backward = target.register_full_backward_hook(lambda module, grad_in, grad_out: gradients.append(grad_out[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        index = int(logits.argmax(1).item()) if class_index is None else class_index
        logits[0, index].backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        heat = torch.relu((weights * activations[0]).sum(1, keepdim=True))
        heat = torch.nn.functional.interpolate(heat, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        heat = heat[0, 0].detach().cpu().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        return heat, index
    finally:
        forward.remove()
        backward.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EfficientNetCBAM(pretrained=False).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    image = Image.open(args.image).convert("RGB")
    size = saved["config"]["image_size"]
    tensor = transform(False, size)(image).unsqueeze(0).to(device)
    heat, index = gradcam(model, tensor)
    base = np.asarray(image.resize((size, size))).astype(np.float32)
    overlay = np.stack([heat * 255, np.zeros_like(heat), (1 - heat) * 100], axis=-1)
    combined = np.uint8(np.clip(0.65 * base + 0.35 * overlay, 0, 255))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(args.output)
    print(f"Predicted class index {index}; Grad-CAM saved to {args.output}")


if __name__ == "__main__":
    main()
