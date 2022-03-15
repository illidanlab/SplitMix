import torch
import numpy as np
from thop import profile
from nets.thop_op_hooks import thop_hooks

# from nets.slimmable_models import Ensemble, EnsembleSubnet, EnsembleGroupSubnet


def count_params_by_state(model):
    """Count #param based on state dict of the given model."""
    if hasattr(model, 'state_size'):  # EnsembleSubnet, EnsembleGroupSubnet
        s = model.state_size()
    else:
        s = 0
        for k, p in model.state_dict().items():
            s = s + p.numel()
    return s


def profile_slimmable_models(model, slim_ratios, verbose=1):
    max_flops = None
    max_params = None
    for slim_ratio in sorted(slim_ratios, reverse=True):
        if hasattr(model, 'switch_slim_mode'):
            model.switch_slim_mode(slim_ratio)
        else:  # if isinstance(model, Ensemble):
            model.set_total_slim_ratio(slim_ratio)
        flops, state_params = profile_model(model, verbose > 1)
        if verbose > 0:
            print(f'slim_ratio: {slim_ratio:.3f} GFLOPS: {flops / 1e9:.4f}, '
                  f'model state size: {state_params / 1e6:.2f}MB')

        if max_flops is None:
            max_flops = flops
            max_params = state_params
        elif verbose > 0:
            print(f"    flop ratio: {flops/max_flops:.3f}, size ratio: {state_params/max_params:.3f},"
                  f" sqrt size ratio: {np.sqrt(state_params/max_params):.3f}")


def profile_model(model, verbose=False, batch_size=2, device='cpu', input_shape=None):
    if input_shape is None:
        input_shape = model.input_shape
    input_shape = (batch_size, *input_shape[1:])
    dummy_input = torch.rand(input_shape).to(device)
    # customized ops:
    #       https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/vision/basic_hooks.py
    state_params = count_params_by_state(model)
    flops, params = profile(model, inputs=(dummy_input,), custom_ops=thop_hooks,
                            verbose=verbose)
    flops = flops / batch_size
    return flops, state_params

