import torch
from torch import Tensor


def ifftn(input: Tensor, dim=None, norm='ortho'):
    x = torch.fft.ifftshift(input, dim=dim)
    x = torch.fft.ifftn(x, dim=dim, norm=norm)
    x = torch.fft.fftshift(x, dim=dim)
    return x

def fftn(input: Tensor, dim=None, norm='ortho'):
    x = torch.fft.ifftshift(input, dim=dim)
    x = torch.fft.fftn(x, dim=dim, norm=norm)
    x = torch.fft.fftshift(x, dim=dim)
    return x
