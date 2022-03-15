"""Ref to HeteroFL pre-activated ResNet18"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from ..models import ScalableModule


hidden_size = [64, 128, 256, 512]


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, norm_layer, conv_layer, scaler):
        super(Block, self).__init__()
        # self.norm_layer = norm_layer
        self.bn1 = norm_layer(in_planes)
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                bias=False)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.scaler = scaler

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_layer(in_planes, self.expansion * planes, kernel_size=1,
                                       stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.scaler(self.shortcut(out)) if hasattr(self, 'shortcut') else x
        out = self.scaler(self.conv1(out))
        out = self.scaler(self.conv2(F.relu(self.bn2(out))))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, norm_layer, conv_layer, scaler):
        super(Bottleneck, self).__init__()
        # self.norm_layer = norm_layer
        self.bn1 = norm_layer(in_planes)
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = norm_layer(planes)
        self.conv3 = conv_layer(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.scaler = scaler

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_layer(in_planes, self.expansion * planes, kernel_size=1,
                                       stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.scaler(self.shortcut(out)) if hasattr(self, 'shortcut') else x
        out = self.scaler(self.conv1(out))
        out = self.scaler(self.conv2(F.relu(self.bn2(out))))
        out = self.scaler(self.conv3(F.relu(self.bn3(out))))
        out += shortcut
        return out


class ResNet(ScalableModule):
    input_shape = [None, 3, 32, 32]

    def __init__(self, hidden_size, block, num_blocks, num_classes=10, bn_type='bn',
                 share_affine=False, track_running_stats=True, width_scale=1.,
                 rescale_init=False, rescale_layer=False):
        super(ResNet, self).__init__(width_scale=width_scale, rescale_init=rescale_init,
                                     rescale_layer=rescale_layer)

        if width_scale != 1.:
            hidden_size = [int(hs * width_scale) for hs in hidden_size]
        self.bn_type = bn_type
        # norm_layer = lambda n_ch: get_bn_layer(bn_type)['2d'](n_ch, track_running_stats=track_running_stats)
        if bn_type == 'bn':
            norm_layer = lambda n_ch: nn.BatchNorm2d(n_ch, track_running_stats=track_running_stats)
        elif bn_type == 'dbn':
            from ..dual_bn import DualNormLayer
            norm_layer = lambda n_ch: DualNormLayer(n_ch, track_running_stats=track_running_stats,
                                                    affine=True, bn_class=nn.BatchNorm2d,
                 share_affine=share_affine)
        else:
            raise RuntimeError(f"Not support bn_type={bn_type}")
        conv_layer = nn.Conv2d

        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(3, hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.bn4 = norm_layer(hidden_size[3] * block.expansion)
        self.linear = nn.Linear(hidden_size[3] * block.expansion, num_classes)

        self.reset_parameters(inp_nonscale_layers=['conv1'])

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, conv_layer):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_layer, conv_layer,
                                self.scaler))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_pre_clf_fea=False):
        out = self.scaler(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn4(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        if return_pre_clf_fea:
            return logits, out
        else:
            return logits

    def print_footprint(self):
        input_shape = self.input_shape
        input_shape[0] = 2
        x = torch.rand(input_shape)
        batch = x.shape[0]
        print(f"input: {np.prod(x.shape[1:])} <= {x.shape[1:]}")
        x = self.conv1(x)
        print(f"conv1: {np.prod(x.shape[1:])} <= {x.shape[1:]}")
        for i_layer, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            print(f"layer {i_layer}: {np.prod(x.shape[1:]):5d} <= {x.shape[1:]}")

def init_param(m):
    """Special init for ResNet"""
    if isinstance(m, (_BatchNorm, _InstanceNorm)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


# Instantiations
def resnet10(**kwargs):
    model = ResNet(hidden_size, Block, [1, 1, 1, 1], **kwargs)
    model.apply(init_param)
    return model


def resnet18(**kwargs):
    model = ResNet(hidden_size, Block, [2, 2, 2, 2], **kwargs)
    model.apply(init_param)
    return model


def resnet26(**kwargs):
    model = ResNet(hidden_size, Block, [3, 3, 3, 3], **kwargs)
    model.apply(init_param)
    return model


def resnet34(**kwargs):
    model = ResNet(hidden_size, Block, [3, 4, 6, 3], **kwargs)
    model.apply(init_param)
    return model


def resnet50(**kwargs):
    model = ResNet(hidden_size, Bottleneck, [3, 4, 6, 3], **kwargs)
    model.apply(init_param)
    return model


def resnet101(**kwargs):
    model = ResNet(hidden_size, Bottleneck, [3, 4, 23, 3], **kwargs)
    model.apply(init_param)
    return model


def resnet152(**kwargs):
    model = ResNet(hidden_size, Bottleneck, [3, 8, 36, 3], **kwargs)
    model.apply(init_param)
    return model


if __name__ == '__main__':
    from nets.profile_func import profile_model

    model = resnet18(track_running_stats=False)
    flops, state_params = profile_model(model, verbose=True)
    print(flops/1e6, state_params/1e6)
