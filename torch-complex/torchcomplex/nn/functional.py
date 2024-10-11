import torch
from torch import Tensor

from torchcomplex import linalg as cLA


def batch_norm(
    input: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    # axes for mean: (0,2,...)
    axes = 0, *range(2, input.dim())
    # tail for view: (1,C,1...)
    tail = 1, input.shape[1], *([1] * (input.dim() - 2))

    if training:
        mean = torch.mean(input, dim=axes)
        if running_mean is not None:
            running_mean += momentum * (mean.detach() - running_mean)
    else:
        if running_mean is None:
            mean = torch.mean(input, dim=axes)
        else:
            mean = running_mean

    centered = input - mean.view(*tail)

    if training:
        V_rr = torch.mul(centered.real, centered.real).mean(dim=axes) + eps
        V_ii = torch.mul(centered.imag, centered.imag).mean(dim=axes) + eps
        V_ri = torch.mul(centered.real, centered.imag).mean(dim=axes)
        if running_var is not None:
            cov = torch.stack([V_rr, V_ri, V_ii])
            running_var += momentum * (cov.detach() - running_var)
    else:
        if running_var is None:
            V_rr = torch.mul(centered.real, centered.real).mean(dim=axes) + eps
            V_ii = torch.mul(centered.imag, centered.imag).mean(dim=axes) + eps
            V_ri = torch.mul(centered.real, centered.imag).mean(dim=axes)
        else:
            V_rr, V_ri, V_ii = running_var

    cov = torch.stack([V_rr, V_ri, V_ri, V_ii], dim=1).view(-1, 2, 2)
    cov_inv_sqroot = cLA.inv_sqroot(cov)
    W_rr = cov_inv_sqroot[..., 0, 0].view(*tail)
    W_ri = cov_inv_sqroot[..., 0, 1].view(*tail)
    W_ii = cov_inv_sqroot[..., 1, 1].view(*tail)
    output = (W_rr * centered.real + W_ri * centered.imag) + 1j * (W_ri * centered.real + W_ii * centered.imag)
    if weight is not None and bias is not None:
        gamma_rr = weight[0, :].view(*tail)
        gamma_ri = weight[1, :].view(*tail)
        gamma_ii = weight[2, :].view(*tail)
        real = gamma_rr * output.real + gamma_ri * output.imag + bias.real.view(*tail)
        imag = gamma_ri * output.real + gamma_ii * output.imag + bias.imag.view(*tail)
        output = real + 1j * imag
    return output


def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5
) -> Tensor:
    # input shape: (N,C,...)
    num_batches = input.shape[0]
    num_channels = input.shape[1]

    # axes for mean: (2,...)
    axes = tuple(range(2, input.dim() + 1))
    # tail for view: (N,G,1...)
    tail = num_batches, num_groups, *([1] * (input.dim() - 1))

    # (N,C,...) -> (N,G,C//G,...)
    input_reshape = input.view(
        num_batches, num_groups, num_channels // num_groups, *tuple(input.shape[2:]))

    # (N,G,C//G,...) -> (N,G,1,...)
    mean = torch.mean(input_reshape, dim=axes, keepdim=True)
    # (N,G,C//G,...)
    centered = input_reshape - mean

    # (N,G,C//G,...) -> (N,G,1,...)
    V_rr = torch.mul(centered.real, centered.real).mean(dim=axes, keepdim=True) + eps
    V_ii = torch.mul(centered.imag, centered.imag).mean(dim=axes, keepdim=True) + eps
    V_ri = torch.mul(centered.real, centered.imag).mean(dim=axes, keepdim=True)

    # 4 * (N,G,1,...) -> (N,G,1,...,2,2)
    cov = torch.stack([V_rr, V_ri, V_ri, V_ii], dim=-1).view(*tail, 2, 2)
    cov_inv_sqroot = cLA.inv_sqroot(cov)

    # (N,G,1,...,2,2) -> 4 * (N,G,1,...)
    W_rr = cov_inv_sqroot[..., 0, 0]
    W_ri = cov_inv_sqroot[..., 0, 1]
    W_ii = cov_inv_sqroot[..., 1, 1]

    # (N,G,C//G,...)
    output = (W_rr * centered.real + W_ri * centered.imag) + 1j * (W_ri * centered.real + W_ii * centered.imag)
    # (N,C,...)
    output = output.view(num_batches, num_channels, *tuple(input.shape[2:]))
    if weight is not None and bias is not None:
        # (1,C,...)
        tail = 1, input.shape[1], *([1] * (input.dim() - 2))
        gamma_rr = weight[0, :].view(*tail)
        gamma_ri = weight[1, :].view(*tail)
        gamma_ii = weight[2, :].view(*tail)

        real = gamma_rr * output.real + gamma_ri * output.imag + bias.real.view(*tail)
        imag = gamma_ri * output.real + gamma_ii * output.imag + bias.imag.view(*tail)
        output = real + 1j * imag
    return output
