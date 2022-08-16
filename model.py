import torch.nn as nn
import torch
from utils import *

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch1x1pool = nn.Sequential(
            nn.AvgPool2d(in_channels, stride=1, padding=0, count_include_pad=False),
            BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1)
        )
        self.branch1x1 = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = torch.cat((self.branch1x1pool(x), self.branch1x1(x), self.branch3x3(x), self.branch3x3stack(x)), dim=1)
        return x

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.branch1x1(x)
        return x

class Spatial_Information_Extraction_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            Inception(3, 16),
            Transition(64, 16),
            Inception(16, 32),
            Transition(128, 32),
            Inception(32, 64),
            Transition(256, 64)
        )

    def forward(self, x):
        x = self.features(x)
        return x