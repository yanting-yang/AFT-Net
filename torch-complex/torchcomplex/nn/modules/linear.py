import torch
from torch import nn, Tensor
from torch.nn import Module

__all__ = ['Linear', 'LazyLinear', 'NaiveLinear']


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.real_layer = nn.Linear(
            in_features, out_features, bias, device, dtype)
        self.imag_layer = nn.Linear(
            in_features, out_features, bias, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        real_part = input.real
        imag_part = input.imag

        real_output = self.real_layer(real_part) - self.imag_layer(imag_part)
        imag_output = self.imag_layer(real_part) + self.real_layer(imag_part)

        return real_output + 1j * imag_output


class LazyLinear(Module):
    def __init__(self, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.real_layer = nn.LazyLinear(out_features, bias, device, dtype)
        self.imag_layer = nn.LazyLinear(out_features, bias, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        real_part = input.real
        imag_part = input.imag

        real_output = self.real_layer(real_part) - self.imag_layer(imag_part)
        imag_output = self.imag_layer(real_part) + self.real_layer(imag_part)

        return real_output + 1j * imag_output


class NaiveLinear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features * 2, bias, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        t = torch.cat((input.real, input.imag), -1)                 # (..., H_in) -> (..., H_in * 2)
        t = self.linear(t)                                          # (..., H_in * 2) -> (..., H_out * 2)
        return torch.view_as_complex(t.view(*t.shape[:-1], -1, 2))  # (..., H_out * 2) -> (..., H_out, 2) -> (..., H_out)
