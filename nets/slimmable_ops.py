"""
Ref: https://github.com/htwang14/CAT/blob/1152f7095d6ea0026c7344b00fefb9f4990444f2/models/FiLM.py#L35
"""
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class SwitchableLayer1D(nn.Module):
    """1-dimensional switchable layer.
    The 1D means the module only requires one dimension variable, like BN.

    Args:
        module_class (nn.Module): Should a module class which takes `num_features`
            as the first arg, and multiple kwargs.
    """
    def __init__(self, module_class, max_num_features: int, slim_ratios: list, **kwargs):
        super(SwitchableLayer1D, self).__init__()
        self.max_num_features = max_num_features
        modules = []
        slim_ratios = sorted(slim_ratios)
        for r in slim_ratios:
            w = int(np.ceil(r * max_num_features))
            modules.append(module_class(w, **kwargs))
        self._switch_modules = nn.ModuleList(modules)
        self.current_module_idx = -1
        self._slim_ratio = max(slim_ratios)
        self.slim_ratios = slim_ratios
        self.ignore_model_profiling = True

    @property
    def slim_ratio(self):
        return self._slim_ratio

    @slim_ratio.setter
    def slim_ratio(self, r):
        self.current_module_idx = self.slim_ratios.index(r)
        self._slim_ratio = r

    def forward(self, x):
        y = self._switch_modules[self.current_module_idx](x)
        return y


class SlimmableOpMixin(object):
    def mix_forward(self, x, mix_num=-1):
        if mix_num < 0:
            mix_num = int(1/self.slim_ratio)
        elif mix_num == 0:
            print("WARNING: not mix anything.")
        out = 0.
        for shift_idx in range(0, mix_num):
            out = out + self._forward_with_partial_weight(x, shift_idx)
        return out * 1. / mix_num

    def _forward_with_partial_weight(self, x, slim_bias_idx, out_slim_bias_idx=None):
        raise NotImplementedError()

    def _compute_slice_bound(self, in_channels, out_channels, slim_bias_idx, out_slim_bias_idx=None):
        out_slim_bias_idx = slim_bias_idx if out_slim_bias_idx is None else out_slim_bias_idx
        out_idx_bias = out_channels * out_slim_bias_idx if not self.non_slimmable_out else 0
        in_idx_bias = in_channels * slim_bias_idx if not self.non_slimmable_in else 0
        return out_idx_bias, (out_idx_bias+out_channels), in_idx_bias, (in_idx_bias+in_channels)


class _SlimmableBatchNorm(_BatchNorm, SlimmableOpMixin):
    """
    BatchNorm2d shared by all sub-networks in slimmable network.
    This won't work according to slimmable net paper.
      See implementation in https://github.com/htwang14/CAT/blob/1152f7095d6ea0026c7344b00fefb9f4990444f2/models/slimmable_ops.py#L28

    If this is used, we will enforce the tracking to be disabled.
    Following https://github.com/dem123456789/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients
    """
    def __init__(self, num_features, eps=1e-5, momentum=None, affine=True,
                 track_running_stats=False, non_slimmable=False):
        assert not track_running_stats, "You should not track stats which cannot be slimmable."
        # if track_running_stats:
        #     assert non_slimmable
        super(_SlimmableBatchNorm, self).__init__(num_features, momentum=momentum, track_running_stats=False, affine=affine, eps=eps)
        self.max_num_features = num_features
        self._slim_ratio = 1.0
        self.slim_bias_idx = 0
        self.out_slim_bias_idx = None
        self.non_slimmable = non_slimmable
        self.mix_forward_num = 1  # 1 means not mix; -1 mix all

    @property
    def slim_ratio(self):
        return self._slim_ratio

    @slim_ratio.setter
    def slim_ratio(self, r):
        self.num_features = self._compute_channels(r)
        self._slim_ratio = r
        if r < 0 and self.track_running_stats:
            raise RuntimeError(f"Try to track state when slim_ratio < 1 is {r}")

    def _compute_channels(self, ratio):
        return self.max_num_features if self.non_slimmable \
            else int(np.ceil(self.max_num_features * ratio))

    def forward(self, x):
        if self.mix_forward_num == 1:
            return self._forward_with_partial_weight(x, self.slim_bias_idx, self.out_slim_bias_idx)
        else:
            return self.mix_forward(x, mix_num=self.mix_forward_num)

    def _forward_with_partial_weight(self, input, slim_bias_idx, out_slim_bias_idx=None):
        out_idx0, out_idx1 = self._compute_slice_bound(self.num_features, slim_bias_idx)
        weight = self.weight[out_idx0:out_idx1]
        bias = self.bias[out_idx0:out_idx1]

        # ----- copy from parent implementation ----
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
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight, bias, bn_training, exponential_average_factor, self.eps)

    def _compute_slice_bound(self, channels, slim_bias_idx):
        idx_bias = channels * slim_bias_idx if not self.non_slimmable else 0
        return idx_bias, (idx_bias+channels)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for name, param in self._parameters.items():
            if param is not None:
                # ------------------------------
                idx_bias = self.num_features * self.slim_bias_idx if not self.non_slimmable else 0
                if name == 'weight':
                    param = param[idx_bias:(idx_bias + self.num_features)]
                elif name == 'bias' and param is not None:
                    param = param[idx_bias:(idx_bias + self.num_features)]
                # ------------------------------
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()


class SlimmableBatchNorm2d(_SlimmableBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class SlimmableBatchNorm1d(_SlimmableBatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

class SlimmableConv2d(nn.Conv2d, SlimmableOpMixin):
    """
    Args:
        non_slimmable_in: Fix the in size
        non_slimmable_out: Fix the out size
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True,
                 non_slimmable_out=False, non_slimmable_in=False,):
        super(SlimmableConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        assert groups == 1, "for now, we can only support single group when slimming."
        assert in_channels > 0
        assert out_channels > 0
        self.max_in_channels = in_channels
        self.max_out_channels = out_channels
        self._slim_ratio = 1.0
        self.slim_bias_idx = 0  # input slim bias idx
        self.out_slim_bias_idx = None  # -1: use the same value as slim_bias_idx
        self.non_slimmable_out = non_slimmable_out
        self.non_slimmable_in = non_slimmable_in
        self.mix_forward_num = -1

    @property
    def slim_ratio(self):
        return self._slim_ratio

    @slim_ratio.setter
    def slim_ratio(self, r):
        self.in_channels, self.out_channels = self._compute_channels(r)
        self._slim_ratio = r

    def _compute_channels(self, ratio):
        in_channels = self.max_in_channels if self.non_slimmable_in \
                else int(np.ceil(self.max_in_channels * ratio))
        out_channels = self.max_out_channels if self.non_slimmable_out \
                else int(np.ceil(self.max_out_channels * ratio))
        return in_channels, out_channels

    def forward(self, x):
        if self.mix_forward_num == 1:
            return self._forward_with_partial_weight(x, self.slim_bias_idx, self.out_slim_bias_idx)
        else:
            return self.mix_forward(x, mix_num=self.mix_forward_num)

    def _forward_with_partial_weight(self, x, slim_bias_idx, out_slim_bias_idx=None):
        out_idx0, out_idx1, in_idx0, in_idx1 = self._compute_slice_bound(
            self.in_channels, self.out_channels, slim_bias_idx, out_slim_bias_idx)
        weight = self.weight[out_idx0:out_idx1, in_idx0:in_idx1]
        bias = self.bias[out_idx0:out_idx1] if self.bias is not None else None
        y = F.conv2d(
            x, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y / self.slim_ratio if self.training and not self.non_slimmable_out else y

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for name, param in self._parameters.items():
            if param is not None:
                # ------------------------------
                out_idx_bias = self.out_channels * self.slim_bias_idx if not self.non_slimmable_out else 0
                if name == 'weight':
                    in_idx_bias = self.in_channels * self.slim_bias_idx \
                        if not self.non_slimmable_in else 0
                    param = param[out_idx_bias:(out_idx_bias+self.out_channels),
                            in_idx_bias:(in_idx_bias+self.in_channels)]
                elif name == 'bias' and param is not None:
                    param = param[out_idx_bias:(out_idx_bias + self.out_channels)]
                # ------------------------------
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()


class SlimmableLinear(nn.Linear, SlimmableOpMixin):
    """
    Args:
        non_slimmable_in: Fix the in size
        non_slimmable_out: Fix the out size
    """
    def __init__(self, in_features: int, out_features: int, bias=True,
                 non_slimmable_out=False, non_slimmable_in=False,):
        super(SlimmableLinear, self).__init__(in_features, out_features, bias=bias)
        self.max_in_features = in_features
        self.max_out_features = out_features
        self._slim_ratio = 1.0
        self.slim_bias_idx = 0  # input slim bias idx
        self.out_slim_bias_idx = None  # -1: use the same value as slim_bias_idx
        self.non_slimmable_out = non_slimmable_out
        self.non_slimmable_in = non_slimmable_in
        self.mix_forward_num = -1

    @property
    def slim_ratio(self):
        return self._slim_ratio

    @slim_ratio.setter
    def slim_ratio(self, r):
        self.in_features, self.out_features = self._compute_channels(r)
        self._slim_ratio = r

    def _compute_channels(self, ratio):
        in_features = self.max_in_features if self.non_slimmable_in \
                else int(np.ceil(self.max_in_features * ratio))
        out_features = self.max_out_features if self.non_slimmable_out \
                else int(np.ceil(self.max_out_features * ratio))
        return in_features, out_features

    def forward(self, x):
        if self.mix_forward_num == 1:
            return self._forward_with_partial_weight(x, self.slim_bias_idx, self.out_slim_bias_idx)
        else:
            return self.mix_forward(x, mix_num=self.mix_forward_num)

    def _forward_with_partial_weight(self, x, slim_bias_idx, out_slim_bias_idx=None):
        out_idx0, out_idx1, in_idx0, in_idx1 = self._compute_slice_bound(
            self.in_features, self.out_features, slim_bias_idx, out_slim_bias_idx)
        weight = self.weight[out_idx0:out_idx1, in_idx0:in_idx1]
        bias = self.bias[out_idx0:out_idx1] if self.bias is not None else None
        out = F.linear(x, weight, bias)
        return out / self.slim_ratio if self.training and not self.non_slimmable_out else out

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for name, param in self._parameters.items():
            if param is not None:
                # ------------------------------
                param = self.get_slim_param(name, param)
                # ------------------------------
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()

    def get_slim_param(self, name, param):
        out_idx_bias = self.out_features * self.slim_bias_idx if not self.non_slimmable_out else 0
        if name == 'weight':
            in_idx_bias = self.in_features * self.slim_bias_idx if not self.non_slimmable_in else 0
            param = param[out_idx_bias:(out_idx_bias + self.out_features),
                    in_idx_bias:(in_idx_bias + self.in_features)]
        elif name == 'bias' and param is not None:
            param = param[out_idx_bias:(out_idx_bias + self.out_features)]
        return param
