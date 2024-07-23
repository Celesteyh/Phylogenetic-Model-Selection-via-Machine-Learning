import torch
import torch.nn as nn


def conv2x1_bn_relu(in_planes: int, out_planes: int, stride: int = 1) -> nn.Sequential:
    """2x1 convolution (same padding) with batch normalization and ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(2, 1), stride=stride, padding='same', bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class Simple_Conv(nn.Module):
    def __init__(self):
        super(Simple_Conv, self).__init__()
        self.conv1 = conv2x1_bn_relu(440, 32)
        self.conv2 = conv2x1_bn_relu(32, 64)
        self.conv3 = conv2x1_bn_relu(64, 96)
        self.conv4 = conv2x1_bn_relu(96, 96)  # B * 96 * 25 * 25

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # B * 96 * 1 * 1
        self.fc = nn.Linear(96, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = Simple_Conv()
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(num_params)

# 64231

