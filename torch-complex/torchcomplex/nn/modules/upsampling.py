from typing import Union, Optional

from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn.common_types import _size_any_t, _ratio_any_t

__all__ = ['Upsample']


class Upsample(Module):
    def __init__(
        self,
        size: _size_any_t | None = None,
        scale_factor: _ratio_any_t | None = None,
        mode: str = 'nearest',
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None
    ) -> None:
        super().__init__()
        self.Upsample = nn.Upsample(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.Upsample(input.real) + 1j * self.Upsample(input.imag)
