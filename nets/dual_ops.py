"""Structure with dual weights."""
import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from typing import Optional


class DualConv2d(nn.Conv2d):
    aux_weight: torch.Tensor
    aux_bias: Optional[torch.Tensor]

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True,
                 fix_out=False, fix_in=False, overlap_rate=0.):
        assert groups == 1, "for now, we can only support single group when slimming."
        if overlap_rate > 0:
            overlap_ch_in = in_channels if fix_in else int((2 - overlap_rate) * in_channels)
            overlap_ch_out = out_channels if fix_out else int((2 - overlap_rate) * out_channels)
            self.conv = super(DualConv2d, self).__init__(
                overlap_ch_in, overlap_ch_out,
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        else:
            self.conv = super(DualConv2d, self).__init__(
                in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias)
            # auxiliary weight, bias
            self.aux_weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, kernel_size, kernel_size))
            if bias:
                self.aux_bias = nn.Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('aux_bias', None)

            init.kaiming_uniform_(self.aux_weight, a=math.sqrt(5))
            if self.aux_bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.aux_bias, -bound, bound)

        self.overlap_rate = overlap_rate
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mode = 0
        self.fix_out = fix_out
        self.fix_in = fix_in

    def forward(self, x):
        if self.overlap_rate > 0:
            in_idx_bias = 0
            out_idx_bias = 0
            if self.mode > 0:
                in_idx_bias = 0 if self.fix_in else int((1 - self.overlap_rate) * self.in_channels)
                out_idx_bias = 0 if self.fix_out else int((1 - self.overlap_rate) * self.out_channels)
            weight = self.weight[out_idx_bias:(out_idx_bias+self.out_channels), in_idx_bias:(in_idx_bias+self.in_channels)]
            bias = self.bias[out_idx_bias:(out_idx_bias + self.out_channels)] if self.bias is not None else None
        else:
            if self.mode > 0:
                weight = self.aux_weight
                bias = self.aux_bias
            else:
                weight = self.weight
                bias = self.bias
        y = F.conv2d(
            x, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y
