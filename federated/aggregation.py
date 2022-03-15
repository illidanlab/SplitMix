"""Methods to aggregate users' model into server and dispatch to users.
"""
import copy
import numpy as np
import torch
from torch import nn


class ClientServerStackModel(nn.Module):
    def __init__(self, client, server):
        super(ClientServerStackModel, self).__init__()
        self.client_encoder = client
        self.server_decoder = server

    def forward(self, x):
        x = self.client_encoder.encode(x)
        return self.server_decoder(x)


class ModelAccumulator(object):
    """Accumulate models. Client models are sequentially trained and accumulatively added to the
    server (w/ weights). At the end of communication, the server model will be divided
    by summed weights.
    If local_bn is True, a dict of bn layers will be kept for all users.

    Concepts:
        running_model: The model used to train. This is not persistent storage. Load by call
            `load_model` at practice.
        server_state_dict: The current state_dict in server.
        accum_state_dict: The accumulated state_dict which will accumulate the trained results
            from running model and update to server_state_dict when fulfilled.

    Args:
        running_model: Model to init state_dict shape and bn layers.
        n_accum: Number of models to accumulate per round. If retrieve before this value,
            an error will raise.
        num_model: Total number of models. Used if local_bn is True.
        local_bn: Whether to keep local bn for all users.
        raise_err_on_early_accum: Raise error if update model when not all users are accumulated.
    """
    def __init__(self, running_model: nn.Module, n_accum, num_model, local_bn=False,
                 raise_err_on_early_accum=True):
        """
        TODO set local_bn to be True for FedRBN, FedBN
        """
        self.n_accum = n_accum
        self._cnt = 0
        self.local_bn = local_bn
        self._weight_sum = 0
        self.raise_err_on_early_accum = raise_err_on_early_accum
        with torch.no_grad():
            self.server_state_dict = {
                k: copy.deepcopy(v) for k, v in running_model.state_dict().items()
            }
            self._accum_state_dict = {
                k: torch.zeros_like(v) for k, v in running_model.state_dict().items()
            }
            if local_bn:
                self.local_state_dict = [{
                    k: copy.deepcopy(v) for k, v in running_model.state_dict().items() if 'bn' in k
                } for _ in range(num_model)]
            else:
                self.local_state_dict = []

    def state_dict(self):
        return {
            'server': self.server_state_dict,
            'clients': self.local_state_dict,
        }

    def load_state_dict(self, state_dict: dict):
        self.server_state_dict = state_dict['server']
        local_state_dict = state_dict['clients']
        if self.local_bn:
            assert len(local_state_dict) > 0, "Not found local state dict when local_bn is set."
            # num_model
            assert len(local_state_dict) == len(self.local_state_dict), \
                f"Load {len(local_state_dict)} local states while expected" \
                f" {len(self.local_state_dict)}"
        else:
            assert len(local_state_dict) == 0, "Found local bn state when local_bn is not set."
        self.local_state_dict = local_state_dict

    def add(self, model_idx, model, weight):
        """Use weight = 1/n_accum to average.
        """
        if self._cnt >= self.n_accum:  # note cnt starts from 0
            raise RuntimeError(f"Try to accumulate {self._cnt}, while only {self.n_accum} models"
                               f" are allowed. Did you forget to reset after accumulated?")
        with torch.no_grad():
            for key in self._accum_state_dict:
                if len(self.local_state_dict) >0 and key in self.local_state_dict[model_idx]:
                    self.local_state_dict[model_idx][key].data.copy_(model.state_dict()[key])
                else:
                    if 'num_batches_tracked' in key:
                        # if self._cnt == 0:
                        # num_batches_tracked is a non trainable LongTensor and
                        # num_batches_tracked are the same for all clients for the given datasets
                        self._accum_state_dict[key].data.copy_(model.state_dict()[key])
                    else:
                        temp = weight * model.state_dict()[key]
                        self._accum_state_dict[key].data.add_(temp)
        self._cnt += 1  # DO THIS at the END such that start from 0.
        self._weight_sum += weight

    @property
    def accumulated_count(self):
        return self._cnt

    @property
    def accum_state_dict(self):
        self.check_full_accum()
        return self._accum_state_dict

    def load_model(self, running_model: nn.Module, model_idx: int, strict=True):
        """Load server model and local BN states into the given running_model."""
        state_dict = {k: v for k, v in self.server_state_dict.items()}
        if len(self.local_state_dict) > 0:
            for k in self.local_state_dict[model_idx]:
                state_dict[k] = self.local_state_dict[model_idx][k]
        running_model.load_state_dict(state_dict, strict=strict)

    def update_server_and_reset(self):
        """Load accumulated state_dict to server_model and
        reset accumulated values but not local bn."""
        self.check_full_accum()
        weight_norm = 1. / self._weight_sum
        with torch.no_grad():
            # update server
            for k in self.server_state_dict:
                if 'num_batches_tracked' in k:
                    self.server_state_dict[k].data.copy_(self._accum_state_dict[k].data)
                else:
                    self.server_state_dict[k].data.copy_(
                        self._accum_state_dict[k].data * weight_norm)

            # reset
            self._cnt = 0
            self._weight_sum = 0
            for k in self._accum_state_dict:
                self._accum_state_dict[k].data.zero_()

    def check_full_accum(self):
        """Check if the number of accumulated models reaches the expected value (n_accum)."""
        if self.raise_err_on_early_accum:
            assert self._cnt == self.n_accum, f"Retrieve before all models are accumulated. " \
                                              f"Expect to accumulate {self.n_accum} but only" \
                                              f" get {self._cnt}"


class SlimmableModelAccumulator(ModelAccumulator):
    """Model accumulate extended for slimmable averaging.
    The class use a model size weight, which enable us to define the weight for each param.
    """
    def __init__(self, running_model: nn.Module, n_accum, num_model, local_bn=False,
                 raise_err_on_early_accum=True):
        super().__init__(running_model, n_accum, num_model, local_bn=local_bn,
                         raise_err_on_early_accum=raise_err_on_early_accum)
        with torch.no_grad():
            # use tensor to define which params are updated and weighted.
            self._weight_sum = {
                k: torch.zeros_like(v) for k, v in running_model.state_dict().items()
            }

    def add(self, model_idx, model, weight, max_slim_ratio=1.0, slim_bias_idx=0,
            out_slim_bias_idx=None):
        """Only add params that are trained (defined by max_slim_ratio). Only weight
        params that are updated.
        Use weight = 1/n_accum to average.
        """
        if self._cnt >= self.n_accum:  # note cnt starts from 0
            raise RuntimeError(f"Try to accumulate {self._cnt}, while only {self.n_accum} models"
                               f" are allowed. Did you forget to reset after accumulated?")
        slim_bias_idxs = slim_bias_idx
        if np.isscalar(slim_bias_idxs):
            slim_bias_idxs = [slim_bias_idxs]
        if len(slim_bias_idxs) > 1:
            assert out_slim_bias_idx is None, "Cannot make 2D-shift."
        with torch.no_grad():
            for slim_bias_idx in slim_bias_idxs:
                # switch to slim mode. then the parameters will be slim in state_dict.
                model.switch_slim_mode(max_slim_ratio, slim_bias_idx=slim_bias_idx, out_slim_bias_idx=out_slim_bias_idx)
                new_state_dict = model.state_dict()
                for key in self._accum_state_dict:
                    if key not in new_state_dict:
                        continue
                    new_tensor = new_state_dict[key]
                    if len(self.local_state_dict) > 0 and key in self.local_state_dict[model_idx]:
                        self.local_state_dict[model_idx][key].data.copy_(new_tensor)
                    else:
                        old_tensor = self._accum_state_dict[key].data
                        if 'num_batches_tracked' in key:
                            # num_batches_tracked is a non trainable LongTensor and
                            # num_batches_tracked are the same for all clients for the given datasets
                            old_tensor.copy_(new_tensor)
                        else:
                            if old_tensor.shape == new_tensor.shape:
                                temp = weight * new_tensor
                                old_tensor.add_(temp)
                                self._weight_sum[key].data.add_(torch.ones_like(new_tensor) * weight)
                            else:
                                # FIXME only can only handle Conv or Linear weights.
                                if len(new_tensor.shape) >= 2:  # conv/linear weights
                                    x, y = new_tensor.shape[:2]
                                    if out_slim_bias_idx is None:
                                        x_bias_idx = slim_bias_idx * x if x < old_tensor.shape[0] else 0
                                    else:
                                        x_bias_idx = out_slim_bias_idx * x if x < old_tensor.shape[0] else 0
                                    y_bias_idx = slim_bias_idx * y if y < old_tensor.shape[1] else 0
                                    sel_old_tensor = old_tensor[x_bias_idx:(x_bias_idx+x), y_bias_idx:(y_bias_idx+y)]
                                    weight_sum = self._weight_sum[key].data[x_bias_idx:(x_bias_idx+x), y_bias_idx:(y_bias_idx+y)]
                                elif len(new_tensor.shape) == 1:  # bias
                                    x = new_tensor.shape[0]
                                    x_bias_idx = slim_bias_idx * x if x < old_tensor.shape[0] else 0
                                    sel_old_tensor = old_tensor[x_bias_idx:(x_bias_idx+x)]
                                    weight_sum = self._weight_sum[key].data[x_bias_idx:(x_bias_idx+x)]
                                else:
                                    raise ValueError(f"Cannot handle the slimmable tensor of dimension at layer {key}: new {new_tensor.shape} and old {sel_old_tensor.shape}")
                                sel_old_tensor.add_(weight * new_tensor)
                                # Update weight of updated params.
                                weight_sum.add_(torch.ones_like(new_tensor) * weight)

        self._cnt += 1  # DO THIS at the END such that start from 0.

    def update_server_and_reset(self):
        """Load accumulated state_dict to server_model and
        reset accumulated values but not local bn."""
        self.check_full_accum()
        with torch.no_grad():
            # update server
            for k in self.server_state_dict:
                if 'num_batches_tracked' in k:
                    self.server_state_dict[k].data.copy_(self._accum_state_dict[k].data)
                else:
                    # Only update server model with those trained parameters.
                    weight_nz_mask = torch.abs(self._weight_sum[k]) > 1e-6  # updated entries
                    self._accum_state_dict[k].data[weight_nz_mask] *= 1. / self._weight_sum[k][weight_nz_mask]  # updated
                    self._accum_state_dict[k].data[~weight_nz_mask] = self.server_state_dict[k].data[~weight_nz_mask]  # not updated
                    self.server_state_dict[k].data.copy_(self._accum_state_dict[k].data)

            # reset
            self._cnt = 0
            for k in self._weight_sum:
                self._weight_sum[k].data.zero_()
            for k in self._accum_state_dict:
                self._accum_state_dict[k].data.zero_()
