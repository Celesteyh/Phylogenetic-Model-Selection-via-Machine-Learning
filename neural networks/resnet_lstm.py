from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    # no padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv2x1_bn_relu(in_planes: int, out_planes: int, stride: int = 1) -> nn.Sequential:
    """2x1 convolution (same padding) with batch normalization and ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(2, 1), stride=stride, padding='same', bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def conv2x2_bn_relu(in_planes: int, out_planes: int, stride: int = 1) -> nn.Sequential:
    """2x1 convolution (same padding) with batch normalization and ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(2, 2), stride=stride, padding='same', bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class BasicBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 7,  # number of base models
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # TODO: change the input size
        self.inplanes = 96  # remember to change this!!!!!!
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = norm_layer(self.inplanes)  # 64 output channels
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 4 new convolutional layers
        # pairwise_conv0 = _conv_bn_relu(filters=32, kernel_size=(2, 1), strides=(1, 1))(input)
        #         pairwise_conv1 = _conv_bn_relu(filters=64, kernel_size=(2, 1), strides=(1, 1))(pairwise_conv0)
        #         pairwise_conv2 = _conv_bn_relu(filters=96, kernel_size=(2, 1), strides=(1, 1))(pairwise_conv1)
        #         pairwise_conv3 = _conv_bn_relu(filters=96, kernel_size=(2, 1), strides=(1, 1))(pairwise_conv2)
        # TODO: channel？
        # H' = (H + 2 * padding - h) / stride + 1
        # input size: 40 * 250 * 48
        self.conv1 = conv2x1_bn_relu(48, 32)  # 100 * 100 * 32
        self.conv2 = conv2x1_bn_relu(32, 64)  # 100 * 100 * 64
        self.conv3 = conv2x1_bn_relu(64, 96)  # 100 * 100 * 96
        self.conv4 = conv2x1_bn_relu(96, 96)  # 100 * 100 * 96
        # layer1 contains 2 or 3 residual blocks without downsample and they both contain convolutional layers
        self.layer1 = self._make_layer(block, 96, layers[0])  # 100 * 100 * 96
        self.layer2 = self._make_layer(block, 192, layers[1], stride=2, dilate=replace_stride_with_dilation[0])  # 50 * 50 * 192
        self.layer3 = self._make_layer(block, 384, layers[2], stride=2, dilate=replace_stride_with_dilation[1])  # 25 * 25 * 384
        self.layer4 = self._make_layer(block, 768, layers[3], stride=2, dilate=replace_stride_with_dilation[2])  # 13 * 13 * 768  实际是 (B, C, H, W) = (B, 768, 13, 13)
        self.lstm = nn.LSTM(input_size=768, hidden_size=960, num_layers=1, batch_first=True, bidirectional=True)  # B * 169 * 1920 => B * (H * W)) * (2 * hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1920, 120)
        self.fc2 = nn.Linear(120, num_classes)

        for m in self.modules():
            # He initialization for convolutional layers
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            # Batch normalization layers are initialized with 1 for weights and 0 for biases
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes)
            )

        layers = []
        # The first block. stride is set to 2 except for the first _make_layer => downsample
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes
        # The second (and third for layer1) block. stride is always set to 1 => no downsample
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 2)  # B * C * H * W => B * C * (H * W) = B * 768 * 169
        x = torch.permute(x, (0, 2, 1))  # B * C * (H * W) => B * (H * W) * C = B * 169 * 768
        x, _ = self.lstm(x)  # x: B * 169 * 1920
        x = x[:, -1, :]  # B * (H * W) * (2 * hidden_size) => B * (2 * hidden_size) = B * 1920
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18_builder() -> ResNet:
    return ResNet(block=BasicBlock, layers=[3, 2, 2, 2])

