import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.modules.batchnorm import _NormBase
from .dual_bn import DualNormLayer


# BN modules
class _MockBatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MockBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return func.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            torch.zeros_like(self.running_mean),
            torch.ones_like(self.running_var),
            self.weight, self.bias, False, exponential_average_factor, self.eps)

class MockBatchNorm1d(_MockBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

class MockBatchNorm2d(_MockBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class BatchNorm2dAgent(nn.BatchNorm2d):
    def __init__(self, *args, log_stat=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_stat = None  # statistic before BN
        self.post_stat = None  # statistic after BN
        self.log_stat = log_stat

    def forward(self, input):
        if not self.log_stat:
            self.pre_stat = None
        else:
            self.pre_stat = {
                'mean': torch.mean(input, dim=[0, 2, 3]).data.cpu().numpy(),
                'var': torch.var(input, dim=[0, 2, 3]).data.cpu().numpy(),
                'data': input.data.cpu().numpy(),
            }
        out = super().forward(input)
        if not self.log_stat:
            self.pre_stat = None
        else:
            self.post_stat = {
                'mean': torch.mean(out, dim=[0,2,3]).data.cpu().numpy(),
                'var': torch.var(out, dim=[0,2,3]).data.cpu().numpy(),
                'data': out.data.cpu().numpy(),
                # 'mean': ((torch.mean(out, dim=[0, 2, 3]) - self.bias)/self.weight).data.cpu().numpy(),
                # 'var': (torch.var(out, dim=[0, 2, 3])/(self.weight**2)).data.cpu().numpy(),
            }
        return out

class BatchNorm1dAgent(nn.BatchNorm1d):
    def __init__(self, *args, log_stat=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_stat = None  # statistic before BN
        self.post_stat = None  # statistic after BN
        self.log_stat = log_stat

    def forward(self, input):
        if not self.log_stat:
            self.pre_stat = None
        else:
            self.pre_stat = {
                'mean': torch.mean(input, dim=[0]).data.cpu().numpy().copy(),
                'var': torch.var(input, dim=[0]).data.cpu().numpy().copy(),
                'data': input.data.cpu().numpy().copy(),
            }
        out = super().forward(input)
        if not self.log_stat:
            self.post_stat = None
        else:
            self.post_stat = {
                'mean': torch.mean(out, dim=[0]).data.cpu().numpy().copy(),
                'var': torch.var(out, dim=[0]).data.cpu().numpy().copy(),
                # 'mean': ((torch.mean(out, dim=[0]) - self.bias)/self.weight).data.cpu().numpy(),
                # 'var': (torch.var(out, dim=[0])/(self.weight**2)).data.cpu().numpy(),
                'data': out.detach().cpu().numpy().copy(),
            }
        # print("post stat mean: ", self.post_stat['mean'])
        return out


def is_film_dual_norm(bn_type: str):
    return bn_type.startswith('fd')


def get_bn_layer(bn_type: str):
    if bn_type.startswith('d'):  # dual norm layer. Example: sbn, sbin, sin
        base_norm_class = get_bn_layer(bn_type[1:])
        bn_class = {
            '1d': lambda num_features, **kwargs: DualNormLayer(num_features, bn_class=base_norm_class['1d'], **kwargs),
            '2d': lambda num_features, **kwargs: DualNormLayer(num_features, bn_class=base_norm_class['2d'], **kwargs),
        }
    elif is_film_dual_norm(bn_type):  # dual norm layer. Example: sbn, sbin, sin
        base_norm_class = get_bn_layer(bn_type[1:])
        bn_class = {
            '1d': lambda num_features, **kwargs: FilmDualNormLayer(num_features, bn_class=base_norm_class['1d'], **kwargs),
            '2d': lambda num_features, **kwargs: FilmDualNormLayer(num_features, bn_class=base_norm_class['2d'], **kwargs),
        }
    elif bn_type == 'bn':
        bn_class = {'1d': nn.BatchNorm1d, '2d': nn.BatchNorm2d}
    elif bn_type == 'none':
        bn_class = {'1d': MockBatchNorm1d,
                    '2d': MockBatchNorm2d}
    else:
        raise ValueError(f"Invalid bn_type: {bn_type}")
    return bn_class
