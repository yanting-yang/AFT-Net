import torch
from torch import nn, Tensor
from torch.nn import Module

from torchcomplex import nn as cnn
import torchmri

class AFT(Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            cnn.Linear(in_features, in_features * 2),
            cnn.LeakyReLU(negative_slope=.1),
            cnn.Linear(in_features * 2, in_features * 2),
            cnn.LeakyReLU(negative_slope=.1),
            cnn.Linear(in_features * 2, in_features)
        )

    def forward(self, input: Tensor) -> Tensor:
        x = torchmri.fft.ifft(input, dim=-2)
        x = self.fc(x)
        return x
