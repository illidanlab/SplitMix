import torch
from torch import nn
from typing import Union
from .dual_ops import DualConv2d


def set_bn_mode(module: nn.Module, is_noised: Union[bool, torch.Tensor]):
    """Set BN mode to be noised or clean. This is only effective for StackedNormLayer
    or DualNormLayer."""

    def set_bn_eval_(m):
        if isinstance(m, (DualNormLayer,)):
            if isinstance(is_noised, torch.Tensor):
                m.clean_input = ~is_noised
            else:
                m.clean_input = not is_noised
        elif isinstance(m, (DualConv2d,)):
            m.mode = 1 if is_noised else 0
    module.apply(set_bn_eval_)


class DualNormLayer(nn.Module):
    """Dual Normalization Layer."""
    _version = 1
    # __constants__ = ['track_running_stats', 'momentum', 'eps',
    #                  'num_features', 'affine']

    def __init__(self, num_features, track_running_stats=True, affine=True, bn_class=None,
                 share_affine=True, **kwargs):
        super(DualNormLayer, self).__init__()
        self.affine = affine
        if bn_class is None:
            bn_class = nn.BatchNorm2d
        self.bn_class = bn_class
        self.share_affine = share_affine
        self.clean_bn = bn_class(num_features, track_running_stats=track_running_stats, affine=self.affine and not self.share_affine, **kwargs)
        self.noise_bn = bn_class(num_features, track_running_stats=track_running_stats, affine=self.affine and not self.share_affine, **kwargs)
        if self.affine and self.share_affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.clean_input = True  # only used in training?

    def reset_parameters(self) -> None:
        if self.affine and self.share_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if isinstance(self.clean_input, bool):
            if self.clean_input:
                out = self.clean_bn(inp)
            else:
                out = self.noise_bn(inp)
        elif isinstance(self.clean_input, torch.Tensor):
            # Separate input. This important at training to avoid mixture of BN stats.
            clean_mask = torch.nonzero(self.clean_input)
            noise_mask = torch.nonzero(~self.clean_input)
            out = torch.zeros_like(inp)

            if len(clean_mask) > 0:
                clean_mask = clean_mask.squeeze(1)
                # print(self.clean_input, clean_mask)
                out_clean = self.clean_bn(inp[clean_mask])
                out[clean_mask] = out_clean
            if len(noise_mask) > 0:
                noise_mask = noise_mask.squeeze(1)
                # print(self.clean_input, noise_mask)
                out_noise = self.noise_bn(inp[noise_mask])
                out[noise_mask] = out_noise
        elif isinstance(self.clean_input, (float, int)):
            assert not self.training, "You should not use both BN at training."
            assert not self.share_affine, "Should not share affine, because we have to use affine" \
                                          " before combination but didn't."
            out_c = self.clean_bn(inp)
            out_n = self.noise_bn(inp)
            out = self.clean_input * 1. * out_c + (1. - self.clean_input) * out_n
        else:
            raise TypeError(f"Invalid self.clean_input: {type(self.clean_input)}")
        if self.affine and self.share_affine:
            # out = F.linear(out, self.weight, self.bias)
            shape = [1] * out.dim()
            shape[1] = -1
            out = out * self.weight.view(*shape) + self.bias.view(*shape)
            assert out.shape == inp.shape
            # TODO how to do the affine?
            # out = F.batch_norm(out, None, None, self.weight, self.bias, self.training)
        return out


class DualBatchNorm2d(DualNormLayer):
    def __init__(self, *args, **kwargs):
        super(DualBatchNorm2d, self).__init__(*args, bn_class=nn.BatchNorm2d, **kwargs)


class DualBatchNorm1d(DualNormLayer):
    def __init__(self, *args, **kwargs):
        super(DualBatchNorm1d, self).__init__(*args, bn_class=nn.BatchNorm1d, **kwargs)


def test():
    norm = DualBatchNorm2d(3)
    norm.eval()
    with torch.no_grad():
        norm.clean_input = torch.randn((32,)) > 0.
        x = torch.randn((32, 3, 2, 2))
        y = norm(x)
        print(list(y.size()))
        assert list(y.size()) == [32, 3, 2, 2]


if __name__ == '__main__':
    test()
    # import doctest
    #
    # doctest.testmod()
