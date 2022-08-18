import torch.nn as nn
import torch
import numpy as np
from utils import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

class Preprocessing(nn.Module):
    def __init__(self, patch_size, overlapping_ratio):
        super().__init__()
        self.kernel = patch_size
        self.ol_ratio = overlapping_ratio
        self.target = None

    def slicing(self, img):
        (h, w) = img.shape[1:]
        lcm = np.lcm(8, self.kernel)
        h_cut, w_cut = (h // lcm) * lcm, (w // lcm) * lcm
        upper_h_cut, left_w_cut = round((h - h_cut) / 2), round((w - w_cut) / 2)
        img = torch.narrow(img, 1, upper_h_cut, h_cut)
        img = torch.narrow(img, 2, left_w_cut, w_cut)
        img = img.unsqueeze(0)

        self.target = img

    def patch_operation(self):

        self.target = self.target.unfold(2, self.kernel, int(self.kernel * self.ol_ratio))
        self.target = self.target.unfold(3, self.kernel, int(self.kernel * self.ol_ratio))

        patches = torch.empty((self.target.shape[2] * self.target.shape[3], 3, self.kernel, self.kernel))
        c = 0
        for i in range(self.target.shape[2]):
            for j in range(self.target.shape[3]):
                patches[c] = self.target[:, :, i, j, :]
                c += 1

        return patches.to(device)


    def forward(self, x):
        self.slicing(x)
        output = self.patch_operation()
        return output

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

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
        self.Upsampling_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Upsampling_4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
        x = self.preprocessing(x)
        feature_2 = self.Sequential_Conv_1(x)
        feature_4 = self.Sequential_Conv_2(feature_2)
        feature_6 = self.Sequential_Conv_3(feature_4)

        return torch.cat((feature_2, self.Upsampling_2(feature_4), self.Upsampling_4(feature_6)), dim=1)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.seperable = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        )

    def forward(self, x):
        return self.seperable(x)

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class Discrimination_Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.SepConvBlock_1 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.SepConvBlock_2 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.SepConvBlock_3 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.SepConvBlock_4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

    def forward(self, x):

        return x

class Model(nn.Module):
    def __init__(self, patch_size, overlapping_ratio):
        super().__init__()
        self.Preprocessing = Preprocessing(patch_size, overlapping_ratio)
        self.SIEN = Spatial_Information_Extraction_Network()
        self.CFEN = Correlation_Feature_Extraction_Network()
        self.DN = Discrimination_Network()

    def forward(self, x):
        x = self.Preprocessing(x)
        SIEN_output = self.SIEN(x)
        CFEN_output = self.CFEN(x)
        x = torch.cat((SIEN_output, CFEN_output), dim=1)
        DN_output = self.DN(x)

