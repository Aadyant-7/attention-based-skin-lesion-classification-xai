"""Convolutional Block Attention Module on spatial feature maps."""
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.mlp(torch.amax(x, dim=(2, 3), keepdim=True)) + self.mlp(torch.mean(x, dim=(2, 3), keepdim=True)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = torch.cat((torch.amax(x, dim=1, keepdim=True), torch.mean(x, dim=1, keepdim=True)), dim=1)
        return x * self.sigmoid(self.conv(features))


class CBAM(nn.Sequential):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__(ChannelAttention(channels, reduction), SpatialAttention(spatial_kernel))
