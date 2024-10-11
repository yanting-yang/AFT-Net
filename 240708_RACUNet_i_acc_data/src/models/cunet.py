from typing import Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import Module

from torchcomplex import nn as cnn


norm_type = 'GroupNorm'
block_type = 'BasicBlock'

__all__ = ['UNet']


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> cnn.Conv2d:
    """3x3 convolution with padding"""
    return cnn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        dilation=dilation,
        groups=groups,
        bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> cnn.Conv2d:
    """1x1 convolution"""
    return cnn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def norm_layer(out_channels: int) -> Union[cnn.BatchNorm2d, cnn.GroupNorm]:
    if norm_type == 'BatchNorm':
        layer = cnn.BatchNorm2d(out_channels)
    elif norm_type == 'GroupNorm':
        layer = cnn.GroupNorm(16, out_channels)
    else:
        raise ValueError(f'norm_type: {norm_type} not supported')
    return layer


class BasicBlock(Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = cnn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = nn.Sequential(
            conv1x1(in_channels, out_channels),
            norm_layer(out_channels)
        )

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


class Bottleneck(Module):
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = cnn.ReLU()
        self.downsample = nn.Sequential(
            conv1x1(in_channels, out_channels),
            norm_layer(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def block_layer(in_channels: int, out_channels: int) -> Union[BasicBlock, Bottleneck]:
    if block_type == 'BasicBlock':
        layer = BasicBlock(in_channels, out_channels)
    elif block_type == 'Bottleneck':
        # type: ignore[assignment]
        layer = Bottleneck(in_channels, out_channels)
    else:
        raise ValueError(f'block_type: {block_type} not supported')
    return layer


class AttentionGate(Module):
    def __init__(self, in_channels, gating_channels, inter_channels) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            cnn.Conv2d(in_channels=gating_channels,
                       out_channels=inter_channels, kernel_size=1)
        )
        self.W_x = nn.Sequential(
            cnn.Conv2d(in_channels=in_channels,
                       out_channels=inter_channels, kernel_size=1)
        )
        self.psi = nn.Sequential(
            cnn.Conv2d(in_channels=inter_channels,
                       out_channels=1, kernel_size=1),
            cnn.Sigmoid()
        )
        self.relu = cnn.ReLU()

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        output_real = torch.mul(x.real, psi.real) - torch.mul(x.imag, psi.imag)
        output_imag = torch.mul(x.real, psi.imag) + torch.mul(x.imag, psi.real)
        return output_real + 1j * output_imag


class Up(Module):
    att: Optional[Module]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention: bool
    ) -> None:
        super().__init__()
        self.up_conv = cnn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.conv = block_layer(out_channels * 2, out_channels)
        if attention:
            self.att = AttentionGate(
                in_channels=out_channels,
                gating_channels=out_channels,
                inter_channels=int(out_channels / 2)
            )
        else:
            self.att = None

    def forward(self, up_x, down_x) -> Tensor:
        up_x = self.up_conv(up_x)
        if self.att is not None:
            x = self.att(down_x, up_x)
            x = torch.cat([up_x, x], dim=1)
        else:
            x = torch.cat([up_x, down_x], dim=1)
        x = self.conv(x)
        return x


class Encoder(Module):
    # Yanting Yang, 2022.04.30
    def __init__(
        self,
        in_channels: int,
        layer_channels: list,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for idx, (in_channels, out_channels) in enumerate(zip([in_channels] + layer_channels[:-1], layer_channels)):
            if idx == 0:
                self.layers.append(
                    block_layer(in_channels, out_channels)
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        cnn.MaxPool2d(kernel_size=2),
                        block_layer(in_channels, out_channels)
                    )
                )

    def forward(self, input: Tensor) -> list:
        output = list()
        x = input
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        return output


class Decoder(Module):
    # Yanting Yang, 2022.04.30
    def __init__(
        self,
        out_channels: int,
        layer_channels: list,
        attention: bool
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for idx, (in_channels, out_channels) in enumerate(zip(layer_channels, [out_channels] + layer_channels[:-1])):
            if idx == 0:
                self.layers.append(
                    conv1x1(in_channels, out_channels)
                )
            else:
                self.layers.append(
                    Up(in_channels, out_channels, attention)
                )

    def forward(self, input: list) -> Tensor:
        output = input[-1]
        for idx, (_input, layer) in reversed(list(enumerate(zip([None] + input[:-1], self.layers)))):
            if idx == 0:
                output = layer(output)
            else:
                output = layer(output, _input)
        return output


class CUNet(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_channels: list,
        attention: bool = False
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, layer_channels)
        self.decoder = Decoder(out_channels, layer_channels, attention)

    def forward(self, input: Tensor) -> Tensor:
        latent = self.encoder(input)
        output = self.decoder(latent)
        return output
