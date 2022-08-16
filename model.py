import torch.nn as nn
import torch
from utils import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
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
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
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
        return torch.cat((self.branch1x1pool(x), self.branch1x1(x), self.branch3x3(x), self.branch3x3stack(x)), dim=1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.branch1x1(x)

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
        return self.features(x)

class CFEN_preprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.first_mask = torch.FloatTensor([[[[0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, -1, 1, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0]]],

                                         [[[0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0],
                                          [0, 1, -2, 1, 0],
                                          [0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0]]],

                                         [[[0, 0, 0, 0, 0],
                                          [0, -1, 2, -1, 0],
                                          [0, 2, -4, 2, 0],
                                          [0, -1, 2, -1, 0],
                                          [0, 0, 0, 0, 0]]],

                                         [[[-1, 2, -2, 2, -1],
                                          [2, -6, 8, -6, 2],
                                          [-2, 8, -12, 8, -2],
                                          [2, -6, 8, -6, 2],
                                          [-1, 2, -2, 2, -1]]]]).to(device)

        self.second_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.second_mask = torch.FloatTensor([[[[0, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0],
                                           [0, 0, -1, 0, 0],
                                           [0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0]]],

                                          [[[0, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0],
                                           [0, 0, -2, 0, 0],
                                           [0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 0]]],

                                          [[[0, 0, 0, 0, 0],
                                           [0, -1, 2, -1, 0],
                                           [0, 2, -4, 2, 0],
                                           [0, -1, 2, -1, 0],
                                           [0, 0, 0, 0, 0]]],

                                          [[[-1, 2, -2, 2, -1],
                                           [2, -6, 8, -6, 2],
                                           [-2, 8, -12, 8, -2],
                                           [2, -6, 8, -6, 2],
                                           [-1, 2, -2, 2, -1]]]]).to(device)

    def forward(self, x):
        x1, x2 = x, x
        for i in range(4):
            self.first_conv.weight.data = self.first_mask[i].repeat(3, 3, 1, 1)
            x1 = self.first_conv(x1)

        for i in range(4):
            self.second_conv.weight.data = self.second_mask[i].repeat(3, 3, 1, 1).to(device)
            x2 = self.second_conv(x2)

        return torch.cat((x1, x2), dim=1)

class Correlation_Feature_Extraction_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocessing = CFEN_preprocess().requires_grad_(False)
        self.Sequential_Conv_1 = nn.Sequential(
            BasicConv2d(6, 16, kernel_size=3, stride=1, padding=1),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1)
        )
        self.Sequential_Conv_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(16, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.Sequential_Conv_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.preprocessing(x)
        feature_2 = self.Sequential_Conv_1(x)
        feature_4 = self.Sequential_Conv_2(feature_2)
        feature_6 = self.Sequential_Conv_3(feature_4)

        return torch.cat((feature_2, feature_4, feature_6), dim=1)