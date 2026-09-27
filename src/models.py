"""EfficientNet-B0 + CBAM + seven-logit progressive-dropout head."""
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / ".cache/torch"))
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from .cbam import CBAM


class EfficientNetCBAM(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super().__init__()
        net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.features = net.features
        self.cbam = CBAM(1280, 16, 7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Hidden dimensions are our design, not specified by Pardede et al.
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(1280, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.cbam(self.features(x))))
