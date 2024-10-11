import torch
from torch import Tensor


def rss(input: Tensor, dim=None, keepdim=False):
    x = input.real ** 2 + input.imag ** 2
    x = torch.sum(x, dim=dim, keepdim=keepdim)
    x = torch.sqrt(x)
    return x
