import torch
import torch.nn as nn


def fc_relu_dropout(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout)
    )


def conv2_bn_relu(in_channels: int, out_channels: int, kernel_size, padding='same'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Conv2d(nn.Module):
    def __init__(self):
        super(Conv2d, self).__init__()
        self.conv2d1 = conv2_bn_relu(1, 16, (4, 1))
        self.conv2d2 = conv2_bn_relu(16, 32, (4, 1))
        self.conv2d3 = conv2_bn_relu(32, 64, (4, 1))
        self.conv2d4 = conv2_bn_relu(64, 128, (4, 1))
        self.conv2d5 = conv2_bn_relu(128, 256, (4, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = fc_relu_dropout(256, 64, 0.2)
        self.fc2 = fc_relu_dropout(64, 16, 0.2)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):  # (40, 50)
        x = x.unsqueeze(1)

        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = self.conv2d4(x)
        x = self.conv2d5(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

# number of parameters = 193171
