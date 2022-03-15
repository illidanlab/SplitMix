import logging
import math
from collections import OrderedDict
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.modules.conv import _ConvNd

from .bn_ops import get_bn_layer
from .dual_bn import DualNormLayer


class BaseModule(nn.Module):
    def set_bn_mode(self, is_noised: Union[bool, torch.Tensor]):
        """Set BN mode to be noised or clean. This is only effective for StackedNormLayer
        or DualNormLayer."""
        def set_bn_eval_(m):
            if isinstance(m, (DualNormLayer,)):
                if isinstance(is_noised, (float, int)):
                    m.clean_input = 1. - is_noised
                elif isinstance(is_noised, torch.Tensor):
                    m.clean_input = ~is_noised
                else:
                    m.clean_input = not is_noised
        self.apply(set_bn_eval_)

    # forward
    def forward(self, x):
        z = self.encode(x)
        logits = self.decode_clf(z)
        return logits

    def encode(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        return z

    def decode_clf(self, z):
        logits = self.classifier(z)
        return logits

    def mix_dual_forward(self, x, lmbd, deep_mix=False):
        if deep_mix:
            self.set_bn_mode(lmbd)
            logit = self.forward(x)
        else:
            # FIXME this will result in unexpected result for non-dual models?
            logit = 0
            if lmbd < 1:
                self.set_bn_mode(False)
                logit = logit + (1 - lmbd) * self.forward(x)

            if lmbd > 0:
                self.set_bn_mode(True)
                logit = logit + lmbd * self.forward(x)
        return logit


def kaiming_uniform_in_(tensor, a=0, mode='fan_in', scale=1., nonlinearity='leaky_relu'):
    """Modified from torch.nn.init.kaiming_uniform_"""
    fan_in = nn.init._calculate_correct_fan(tensor, mode)
    fan_in *= scale
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def scale_init_param(m, scale_in=1.):
    """Scale w.r.t. input dim."""
    if isinstance(m, (nn.Linear, _ConvNd)):
        kaiming_uniform_in_(m.weight, a=math.sqrt(5), scale=scale_in, mode='fan_in')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            fan_in *= scale_in
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    return m


class Scaler(nn.Module):
    def __init__(self, width_scale):
        super(Scaler, self).__init__()
        self.width_scale = width_scale

    def forward(self, x):
        return x / self.width_scale if self.training else x


class ScalableModule(BaseModule):
    def __init__(self, width_scale=1., rescale_init=False, rescale_layer=False):
        super(ScalableModule, self).__init__()
        if rescale_layer:
            self.scaler = Scaler(width_scale)
        else:
            self.scaler = nn.Identity()
        self.rescale_init = rescale_init
        self.width_scale = width_scale

    def reset_parameters(self, inp_nonscale_layers):
        if self.rescale_init and self.width_scale != 1.:
            for name, m in self._modules.items():
                if name not in inp_nonscale_layers:  # NOTE ignore the layer with non-slimmable inp.
                    m.apply(lambda _m: scale_init_param(_m, scale_in=1./self.width_scale))

    @property
    def rescale_layer(self):
        return not isinstance(self.scaler, nn.Identity)

    @rescale_layer.setter
    def rescale_layer(self, enable=True):
        if enable:
            self.scaler = Scaler(self.width_scale)
        else:
            self.scaler = nn.Identity()


class DigitModel(ScalableModule):
    """
    Model for benchmark experiment on Digits. 
    """
    input_shape = [None, 3, 28, 28]

    def __init__(self, num_classes=10, bn_type='bn', track_running_stats=True,
                 width_scale=1., share_affine=True, rescale_init=False, rescale_layer=False):
        super(DigitModel, self).__init__(width_scale=width_scale, rescale_init=rescale_init,
                                         rescale_layer=rescale_layer)
        bn_class = get_bn_layer(bn_type)
        bn_kwargs = dict(
            track_running_stats=track_running_stats,
        )
        if bn_type.startswith('d'):  # dual BN
            bn_kwargs['share_affine'] = share_affine
        conv_layers = [64, 64, 128]
        fc_layers = [2048, 512]
        conv_layers = [int(width_scale*l) for l in conv_layers]
        fc_layers = [int(width_scale*l) for l in fc_layers]
        self.bn_type = bn_type

        self.conv1 = nn.Conv2d(3, conv_layers[0], 5, 1, 2)
        self.bn1 = bn_class['2d'](conv_layers[0], **bn_kwargs)

        self.conv2 = nn.Conv2d(conv_layers[0], conv_layers[1], 5, 1, 2)
        self.bn2 = bn_class['2d'](conv_layers[1], **bn_kwargs)

        self.conv3 = nn.Conv2d(conv_layers[1], conv_layers[2], 5, 1, 2)
        self.bn3 = bn_class['2d'](conv_layers[2], **bn_kwargs)
    
        self.fc1 = nn.Linear(conv_layers[2]*7*7, fc_layers[0])
        self.bn4 = bn_class['1d'](fc_layers[0], **bn_kwargs)

        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.bn5 = bn_class['1d'](fc_layers[1], **bn_kwargs)

        self.fc3 = nn.Linear(fc_layers[1], num_classes)

        self.reset_parameters(inp_nonscale_layers=['conv1'])

    def forward(self, x):
        z = self.encode(x)
        return self.decode_clf(z)

    def encode(self, x):
        x = func.relu(self.bn1(self.scaler(self.conv1(x))))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.scaler(self.conv2(x))))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.scaler(self.conv3(x))))

        x = x.view(x.shape[0], -1)
        return x

    def decode_clf(self, x):
        x = self.scaler(self.fc1(x))
        x = self.bn4(x)
        x = func.relu(x)

        x = self.scaler(self.fc2(x))
        x = self.bn5(x)
        x = func.relu(x)

        logits = self.fc3(x)
        return logits


class AlexNet(ScalableModule):
    """
    used for DomainNet and Office-Caltech10
    """
    input_shape = [None, 3, 256, 256]

    def load_state_dict(self, state_dict, strict: bool = True):
        legacy_keys = []
        for key in state_dict:
            if 'noise_disc' in key:
                legacy_keys.append(key)
        if len(legacy_keys) > 0:
            logging.debug(f"Found old version of AlexNet. Ignore {len(legacy_keys)} legacy"
                          f" keys: {legacy_keys}")
            for key in legacy_keys:
                state_dict.pop(key)
        return super().load_state_dict(state_dict, strict)

    def __init__(self, num_classes=10, track_running_stats=True, bn_type='bn', share_affine=True,
                 width_scale=1., rescale_init=False, rescale_layer=False):
        super(AlexNet, self).__init__(width_scale=width_scale, rescale_init=rescale_init,
                                      rescale_layer=rescale_layer)
        self.bn_type = bn_type
        bn_class = get_bn_layer(bn_type)
        # share_affine
        bn_kwargs = dict(
            track_running_stats=track_running_stats,
        )
        if bn_type.startswith('d'):  # dual BN
            bn_kwargs['share_affine'] = share_affine
        plus_layer_i = 0
        feature_layers = []
        feature_layers += [
            ('conv1', nn.Conv2d(3, int(width_scale*64), kernel_size=11, stride=4, padding=2)),
            ('scaler1', self.scaler),
            ('bn1', bn_class['2d'](int(width_scale*64), **bn_kwargs)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv2', nn.Conv2d(int(width_scale*64), int(width_scale*192), kernel_size=5, padding=2)),
            ('scaler2', self.scaler),
            ('bn2', bn_class['2d'](int(width_scale*192), **bn_kwargs)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv3', nn.Conv2d(int(width_scale*192), int(width_scale*384), kernel_size=3, padding=1)),
            ('scaler3', self.scaler),
            ('bn3', bn_class['2d'](int(width_scale*384), **bn_kwargs)),
            ('relu3', nn.ReLU(inplace=True)),

            ('conv4', nn.Conv2d(int(width_scale*384), int(width_scale*256), kernel_size=3, padding=1)),
            ('scaler4', self.scaler),
            ('bn4', bn_class['2d'](int(width_scale*256), **bn_kwargs)),
            ('relu4', nn.ReLU(inplace=True)),

            ('conv5', nn.Conv2d(int(width_scale*256), int(width_scale*256), kernel_size=3, padding=1)),
            ('scaler5', self.scaler),
            ('bn5', bn_class['2d'](int(width_scale*256), **bn_kwargs)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]
        self.features = nn.Sequential(OrderedDict(feature_layers))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(int(width_scale*256) * 6 * 6, int(width_scale*4096))),
                ('scaler6', self.scaler),
                ('bn6', bn_class['1d'](int(width_scale*4096), **bn_kwargs)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(int(width_scale*4096), int(width_scale*4096))),
                ('scaler7', self.scaler),
                ('bn7', bn_class['1d'](int(width_scale*4096), **bn_kwargs)),
                ('relu7', nn.ReLU(inplace=True)),

                ('fc3', nn.Linear(int(width_scale*4096), num_classes)),
            ])
        )
        self.reset_parameters(inp_nonscale_layers=[])
        if self.rescale_init and self.width_scale != 1.:
            self.features.conv1.reset_parameters()  # ignore rescale init

    def forward(self, x):
        z = self.encode(x)
        logits = self.decode_clf(z)
        return logits

    def encode(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        return z

    def decode_clf(self, z):
        logits = self.classifier(z)
        return logits


if __name__ == '__main__':
    from nets.profile_func import profile_model, count_params_by_state

    model = AlexNet(width_scale=1., depth_plus=0)
    fea_params = count_params_by_state(model.features)
    clf_params = count_params_by_state(model.classifier)
    print(f"fea_params {fea_params/1e6} MB, clf_params: {clf_params/1e6} MB")
    for width_scale in [0.125]:  # , 1.0]:  # , 0.25, 0.5, 1.0]:
        for depth_plus in [0, 4, 8, 16, 22, 32, 256]:
            model = AlexNet(width_scale=width_scale, depth_plus=depth_plus)
            flops, state_params = profile_model(model)
            print(f' {width_scale:.3f}xWide {depth_plus}+Dep | GFLOPS {flops / 1e9:.4f}, '
                  f'model state size: {state_params / 1e6:.2f}MB')
            n_nets = int(1/width_scale)
            print(f"      {n_nets}xNets | GFLOPS {n_nets*flops / 1e9:.4f}, "
                  f"model state size: {n_nets*state_params / 1e6:.2f}MB")
    for width_scale in [1.]:  # , 1.0]:  # , 0.25, 0.5, 1.0]:
        for depth_plus in [0]:
            model = AlexNet(width_scale=width_scale, depth_plus=depth_plus)
            flops, state_params = profile_model(model)
            print(f' {width_scale:.3f}xWide {depth_plus}+Dep | GFLOPS {flops / 1e9:.4f}, '
                  f'model state size: {state_params / 1e6:.2f}MB')
    print(model)
