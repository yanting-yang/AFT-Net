import torch
from torch import Tensor
from torch.nn import Module, init
from torch.nn.parameter import Parameter

from .. import functional as cF

__all__ = ['GroupNorm']


class GroupNorm(Module):
    def __init__(
            self,
            num_groups: int,
            num_channels: int,
            eps: float = 1e-5,
            affine: bool = True,
            device=None,
            dtype=None
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(
                3, num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(
                num_channels, device=device, dtype=torch.cfloat))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            self.weight.data[0, :] = torch.tensor(0.5).sqrt()
            self.weight.data[1, :] = torch.tensor(0)
            self.weight.data[2, :] = torch.tensor(0.5).sqrt()
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return cF.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
