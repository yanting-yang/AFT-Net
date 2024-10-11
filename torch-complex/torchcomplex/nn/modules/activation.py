import warnings

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module

from torch.nn.parameter import Parameter

__all__ = ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'PReLU', 'modReLU', 'Siglog']


class ReLU(Module):
    def __init__(self, inplace: bool = False) -> None:
        if inplace:
            warnings.warn('inplace option is ignored in complex ReLU')
        super().__init__()
        self.inplace = False

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input.real, self.inplace) + 1j * F.relu(input.imag, self.inplace)


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(input.real) + 1j * torch.sigmoid(input.imag)


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.tanh(input.real) + 1j * torch.tanh(input.imag)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        if inplace:
            warnings.warn('inplace option is ignored in complex LeakyReLU')
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = False

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input.real, self.negative_slope, self.inplace) + 1j * F.leaky_relu(input.imag, self.negative_slope, self.inplace)


class PReLU(Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        self.weight = Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.init)

    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input.real, self.weight) + 1j * F.prelu(input.imag, self.weight)


class modReLU(Module):
    """@InProceedings{pmlr-v48-arjovsky16,
    title = {Unitary Evolution Recurrent Neural Networks},
    author = {Arjovsky, Martin and Shah, Amar and Bengio, Yoshua},
    booktitle = {Proceedings of The 33rd International Conference on Machine Learning},
    series = {Proceedings of Machine Learning Research},
    publisher = {PMLR},
    year = {2016},
    month = {Jun},
    url = {https://proceedings.mlr.press/v48/arjovsky16.html}
    }"""

    def __init__(self) -> None:
        super().__init__()
        self.bias = Parameter(torch.rand(1) * 0.25)

    def forward(self, input: Tensor) -> Tensor:
        mag = torch.abs(input)
        return F.relu(mag + self.bias) * input / mag


class Siglog(Module):
    """@phdthesis{Virtue:EECS-2019-126,
    title = {Complex-valued Deep Learning with Applications to Magnetic Resonance Image Synthesis},
    author = {Virtue, Patrick},
    school = {EECS Department, University of California, Berkeley},
    year = {2019},
    month = {Aug},
    url = {http://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-126.html}
    }"""

    def forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(torch.log(torch.abs(input))) * torch.exp(1j * torch.angle(input))
