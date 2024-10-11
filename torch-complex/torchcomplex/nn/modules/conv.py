from typing import Union

from torch import nn, Tensor
from torch.nn import Module
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']


class _ConvNd(Module):
    real_conv: Module
    imag_conv: Module

    def forward(self, input: Tensor) -> Tensor:
        real_part = input.real
        imag_part = input.imag

        real_conv_output = self.real_conv(real_part) - self.imag_conv(imag_part)
        imag_conv_output = self.imag_conv(real_part) + self.real_conv(imag_part)

        return real_conv_output + 1j * imag_conv_output


class Conv1d(_ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: _size_1_t | str = 0,
            dilation: _size_1_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        super().__init__()
        self.real_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        self.imag_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )


class Conv2d(_ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t | str = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        super().__init__()
        self.real_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        self.imag_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )


class Conv3d(_ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_3_t,
            stride: _size_3_t = 1,
            padding: _size_3_t | str = 0,
            dilation: _size_3_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        super().__init__()
        self.real_conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        self.imag_conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )


class ConvTranspose1d(_ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: _size_1_t = 0,
            output_padding: _size_1_t = 0,
            groups: int = 1,
            bias: bool = True,
            dilation: _size_1_t = 1,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        super().__init__()
        self.real_conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, device, dtype
        )
        self.imag_conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, device, dtype
        )


class ConvTranspose2d(_ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            output_padding: _size_2_t = 0,
            groups: int = 1,
            bias: bool = True,
            dilation: _size_2_t = 1,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        super().__init__()
        self.real_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, device, dtype
        )
        self.imag_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, device, dtype
        )


class ConvTranspose3d(_ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_3_t,
            stride: _size_3_t = 1,
            padding: _size_3_t = 0,
            output_padding: _size_3_t = 0,
            groups: int = 1,
            bias: bool = True,
            dilation: _size_3_t = 1,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        super().__init__()
        self.real_conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, device, dtype
        )
        self.imag_conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, device, dtype
        )
