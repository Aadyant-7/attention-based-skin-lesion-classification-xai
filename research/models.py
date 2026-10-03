"""Common seven-class CNN interface for future paired backbone/CBAM studies."""
import os
from pathlib import Path
os.environ.setdefault('TORCH_HOME',str(Path(__file__).resolve().parents[1]/'.cache/torch'))
from torch import nn
from torchvision import models
from src.cbam import CBAM

CHANNELS={'efficientnet_b0':1280,'efficientnet_b2':1408,'resnet50':2048,
          'densenet121':1024,'mobilenet_v3_large':960,'convnext_tiny':768}


class ResearchClassifier(nn.Module):
    def __init__(self, backbone, weights=None, attention='none', dropout=.2):
        super().__init__()
        if backbone not in CHANNELS or attention not in ('none','cbam'):
            raise ValueError('Unsupported backbone/attention')
        if isinstance(weights,str):
            enum=models.get_model_weights(backbone)
            name=weights.rsplit('.',1)[-1]
            if name=='DEFAULT':
                raise ValueError('Record an explicit pretrained weight version, not DEFAULT')
            if '.' in weights and weights.split('.')[0]!=enum.__name__:
                raise ValueError('Weights belong to a different backbone')
            weights=enum[name]
        net=models.get_model(backbone,weights=weights)
        if backbone=='resnet50':
            self.features=nn.Sequential(*list(net.children())[:-2])
        elif backbone=='densenet121':
            self.features=nn.Sequential(net.features,nn.ReLU(inplace=False))
        else:
            self.features=net.features
        channels=CHANNELS[backbone]
        self.attention=CBAM(channels,16,7) if attention=='cbam' else nn.Identity()
        # ConvNeXt uses channel LayerNorm after pooling; preserve that operation.
        norm=net.classifier[0] if backbone=='convnext_tiny' else nn.Identity()
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),norm,nn.Flatten(),nn.Dropout(dropout),nn.Linear(channels,7))
        self.backbone_name=backbone

    def forward(self,x):
        return self.head(self.attention(self.features(x)))
