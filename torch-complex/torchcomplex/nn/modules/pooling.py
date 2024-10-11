import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t

from typing import Optional

__all__ = ['MaxPool1d', 'MaxPool2d', 'MaxPool3d']


class _MaxPoolNd(Module):
    def __init__(
            self,
            kernel_size: _size_any_t,
            stride: _size_any_t | None = None,
            padding: _size_any_t = 0,
            dilation: _size_any_t = 1,
            return_indices: bool = False,
            ceil_mode: bool = False
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode


class MaxPool1d(_MaxPoolNd):
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    dilation: _size_1_t

    def forward(self, input: Tensor) -> Tensor:
        magnitude = torch.abs(input)
        _, indices = F.max_pool1d(
            magnitude,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            return_indices=True,
            ceil_mode=self.ceil_mode
        )
        flattened = torch.flatten(input, start_dim=-1)
        output = torch.gather(flattened, dim=-1, index=torch.flatten(indices, start_dim=-1))
        return output.reshape(indices.shape)


class MaxPool2d(_MaxPoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, input: Tensor) -> Tensor:
        magnitude = torch.abs(input)
        _, indices = F.max_pool2d(
            magnitude,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            return_indices=True,
            ceil_mode=self.ceil_mode
        )
        flattened = torch.flatten(input, start_dim=-2)
        output = torch.gather(flattened, dim=-1, index=torch.flatten(indices, start_dim=-2))
        return output.reshape(indices.shape)


class MaxPool3d(_MaxPoolNd):
    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    dilation: _size_3_t

    def forward(self, input: Tensor) -> Tensor:
        magnitude = torch.abs(input)
        _, indices = F.max_pool3d(
            magnitude,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            return_indices=True,
            ceil_mode=self.ceil_mode
        )
        flattened = torch.flatten(input, start_dim=-3)
        output = torch.gather(flattened, dim=-1, index=torch.flatten(indices, start_dim=-3))
        return output.reshape(indices.shape)
