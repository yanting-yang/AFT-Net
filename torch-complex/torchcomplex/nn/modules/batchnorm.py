from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import init, Module
from torch.nn.parameter import Parameter

from .. import functional as cF

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d']


class _NormBase(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.empty(3, num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, dtype=torch.cfloat))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                'running_mean',
                torch.zeros(num_features, dtype=torch.cfloat, **{k: v for k, v in factory_kwargs.items() if k != 'dtype'})
            )
            self.register_buffer('running_var', torch.ones(3, num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer(
                'num_batches_tracked',
                torch.tensor(0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != 'dtype'})
            )
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            self.weight.data[0, :] = torch.tensor(0.5).sqrt()
            self.weight.data[1, :] = torch.tensor(0)
            self.weight.data[2, :] = torch.tensor(0.5).sqrt()
            init.zeros_(self.bias)

    def _check_input_dim(self, input: Tensor) -> None:
        raise NotImplementedError


class _BatchNorm(_NormBase):
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return cF.batch_norm(
            input,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps
        )


class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 4:
            raise ValueError(
                "expected 4D input (got {}D input)".format(input.dim())
            )


class BatchNorm3d(_BatchNorm):
    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 5:
            raise ValueError(
                "expected 5D input (got {}D input)".format(input.dim())
            )
