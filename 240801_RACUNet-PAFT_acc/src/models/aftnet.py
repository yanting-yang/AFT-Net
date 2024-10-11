import torch
from torch import nn, Tensor
from torch.nn import Module

from torchcomplex import nn as cnn
import torchmri
from .cunet import CUNet


class PAFT_DC(Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            cnn.Linear(in_features, in_features * 2),
            cnn.LeakyReLU(negative_slope=.1),
            cnn.Linear(in_features * 2, in_features * 2),
            cnn.LeakyReLU(negative_slope=.1),
            cnn.Linear(in_features * 2, in_features)
        )

    def forward(self, ksp_in: Tensor) -> Tensor:
        x = torchmri.fft.ifftn(ksp_in, dim=-2)
        isp_pred = self.fc(x)
        ksp_pred = torchmri.fft.fftn(isp_pred, dim=(-2, -1))
        ksp_pred = apply_k_space_consistency(ksp_in, ksp_pred)
        isp_pred = torchmri.fft.ifftn(ksp_pred, dim=(-2, -1))
        return isp_pred


def apply_k_space_consistency(k_space_under_gt, k_space_output):
    """
    Apply k-space data consistency by replacing the values in the k-space output
    with the undersampled ground truth values where the ground truth is not zero.

    Args:
        k_space_gt (torch.Tensor): The undersampled ground truth k-space data.
        k_space_output (torch.Tensor): The output k-space data from the model.

    Returns:
        torch.Tensor: The k-space output with data consistency applied.
    """
    # Ensure the tensors are on the same device
    k_space_under_gt = k_space_under_gt.to(k_space_output.device)

    # Create a mask where the undersampled ground truth is not zero
    mask = k_space_under_gt != 0

    # Apply the mask to update the k-space output with the ground truth values
    k_space_output[mask] = k_space_under_gt[mask]

    return k_space_output


class RACUNet_I(Module):
    def __init__(self) -> None:
        super().__init__()
        self.acunet = CUNet(
            in_channels=4,
            out_channels=4,
            layer_channels=[32, 64, 128, 256, 512],
            attention=True
        )

    def forward(self, isp_in: Tensor, ksp_in: Tensor) -> Tensor:
        isp_pred = isp_in + self.acunet(isp_in)
        ksp_pred = torchmri.fft.fftn(isp_pred, dim=(-2, -1))
        ksp_pred = apply_k_space_consistency(ksp_in, ksp_pred)
        isp_pred = torchmri.fft.ifftn(ksp_pred, dim=(-2, -1))
        return isp_pred

class PAFT_RACUNet(Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.aft = PAFT_DC(in_features)
        self.post_unet = RACUNet_I()
