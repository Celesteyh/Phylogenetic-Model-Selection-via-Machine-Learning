# class SqueezeExcitation is copied from PyTorch implementation of Squeeze-and-Excitation Networks
# See LICENSE file for details

import torch
import torch.nn as nn


def conv1x1_bn_relu(in_channels: int, out_channels: int, padding='same', kernel_size=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class SqueezeExcitation(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation=torch.nn.ReLU,
        scale_activation=torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input


class Conv_SEB(nn.Module):
    def __init__(self):
        super(Conv_SEB, self).__init__()
        self.conv1 = conv1x1_bn_relu(440, 32)
        self.se1 = SqueezeExcitation(32, 16)
        self.conv2 = conv1x1_bn_relu(32, 64)
        self.se2 = SqueezeExcitation(64, 16)
        self.conv3 = conv1x1_bn_relu(64, 96)
        self.se3 = SqueezeExcitation(96, 16)
        self.conv4 = conv1x1_bn_relu(96, 96)
        self.se4 = SqueezeExcitation(96, 16)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, 7)

    def forward(self, x):  # (440, 25, 25)
        x = self.conv1(x)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.se2(x)
        x = self.conv3(x)
        x = self.se3(x)
        x = self.conv4(x)
        x = self.se4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# number of parameters = 42311

