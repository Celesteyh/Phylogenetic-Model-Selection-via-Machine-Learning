import torch
import torch.nn as nn


def conv2x1_bn_relu(in_planes: int, out_planes: int, stride: int = 1) -> nn.Sequential:
    """2x1 convolution (same padding) with batch normalization and ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(2, 1), stride=stride, padding='same', bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class Simple_Attention1(nn.Module):
    def __init__(self):
        super(Simple_Attention1, self).__init__()
        self.conv1 = conv2x1_bn_relu(440, 32)
        self.conv2 = conv2x1_bn_relu(32, 64)
        self.conv3 = conv2x1_bn_relu(64, 96)
        self.conv4 = conv2x1_bn_relu(96, 96)  # B * 96 * 25 * 25

        self.attention = nn.MultiheadAttention(embed_dim=96, num_heads=8, batch_first=True)  # B * 625 * 96
        self.fc = nn.Linear(96, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # reshape x to (B, 625, 96)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(3))

        x = self.attention(x, x, x)[0]
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = Simple_Attention1()
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(num_params)

# 101479

