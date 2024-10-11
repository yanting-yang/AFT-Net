import torch
from torch import Tensor


def sqroot(M: Tensor) -> Tensor:
    """
    Compute the square root of a square matrix.
    M = R^2
    """
    tail = *tuple(M.shape[:-2]), 1, 1
    A = M[..., 0, 0]
    B = M[..., 0, 1]
    C = M[..., 1, 0]
    D = M[..., 1, 1]

    s = torch.sqrt(A * D - B * C)
    t = torch.sqrt(A + D + 2 * s)
    I = torch.ones_like(M) * torch.eye(2)
    R = (M + s.view(*tail) * I) / t.view(*tail)
    return R


def inv_sqroot(input: Tensor) -> Tensor:
    A = input[..., 0, 0]
    B = input[..., 0, 1]
    C = input[..., 1, 0]
    D = input[..., 1, 1]

    s = torch.sqrt(A * D - B * C)
    t = torch.sqrt(A + D + 2 * s)
    denom = s * t

    a, b = (D + s) / denom, -B / denom
    c, d = -C / denom, (A + s) / denom

    output = torch.stack([a, b, c, d], dim=-1).view(*tuple(input.shape[:-2]), 2, 2)

    return output
