import copy, argparse
import numpy as np
import math
from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed=None):
    import random
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    def extend(self, items):
        self.values.extend(items)
        self.counter += len(items)

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        values = self.values
        if len(values) > 0:
            return ','.join([f" {metric}: {eval(f'np.{metric}')(values)}"
                             for metric in ['mean', 'std', 'min', 'max']])
        else:
            return 'empy meter'

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class LocalMaskCrossEntropyLoss(nn.CrossEntropyLoss):
    """Should be used for class-wise non-iid.
    Refer to HeteroFL (https://openreview.net/forum?id=TNkPBBYFkXg)
    """
    def __init__(self, num_classes, **kwargs):
        super(LocalMaskCrossEntropyLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        
    def forward(self, input, target):
        classes = torch.unique(target)
        mask = torch.zeros_like(input)
        for c in range(self.num_classes):
            if c in classes:
                mask[:, c] = 1  # select included classes
        return F.cross_entropy(input*mask, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


# ///////////// samplers /////////////
class _Sampler(object):
    def __init__(self, arr):
        self.arr = copy.deepcopy(arr)

    def next(self):
        raise NotImplementedError()


class shuffle_sampler(_Sampler):
    def __init__(self, arr, rng=None):
        super().__init__(arr)
        if rng is None:
            rng = np.random
        rng.shuffle(self.arr)
        self._idx = 0
        self._max_idx = len(self.arr)

    def next(self):
        if self._idx >= self._max_idx:
            np.random.shuffle(self.arr)
            self._idx = 0
        v = self.arr[self._idx]
        self._idx += 1
        return v


class random_sampler(_Sampler):
    def next(self):
        # np.random.randint(0, int(1 / slim_ratios[0]))
        v = np.random.choice(self.arr)  # single value. If multiple value, note the replace param.
        return v


class constant_sampler(_Sampler):
    def __init__(self, value):
        super().__init__([])
        self.value = value

    def next(self):
        return self.value


# ///////////// lr schedulers /////////////
class CosineAnnealingLR(object):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, T_max, eta_max=1e-2, eta_min=0, last_epoch=0, warmup=None):
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self._cur_lr = eta_max
        self._eta_max = eta_max
        # super(CosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)
        self.warmup = warmup

    def step(self):
        self._cur_lr = self._get_lr()
        self.last_epoch += 1
        return self._cur_lr

    def _get_lr(self):
        if self.warmup is not None and self.warmup > 0:
            if self.last_epoch < self.warmup:
                return self._eta_max * ((self.last_epoch+1e-2) / self.warmup)
            elif self.last_epoch == self.warmup:
                return self._eta_max
        if self.last_epoch == 0:
            return self.eta_max
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self._cur_lr + (self.eta_max - self.eta_min) * \
                    (1 - math.cos(math.pi / self.T_max)) / 2
        return (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / \
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * \
                (self._cur_lr - self.eta_min) + self.eta_min


class MultiStepLR(object):
    def __init__(self, eta_max, milestones, gamma=0.1, last_epoch=-1, warmup=None):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.last_epoch = last_epoch
        self._cur_lr = eta_max
        self._eta_max = eta_max
        # super(MultiStepLR, self).__init__(optimizer, last_epoch, verbose)
        self.warmup = warmup

    def step(self):
        self._cur_lr = self._get_lr()
        self.last_epoch += 1
        return self._cur_lr

    def _get_lr(self):
        if self.warmup is not None and self.warmup > 0:
            if self.last_epoch < self.warmup:
                return self._eta_max * ((self.last_epoch+1e-3) / self.warmup)
            elif self.last_epoch == self.warmup:
                return self._eta_max
        if self.last_epoch not in self.milestones:
            return self._cur_lr
        return self._cur_lr * self.gamma ** self.milestones[self.last_epoch]


def test_lr_sch(sch_name='cos'):
    lr_init = 0.1
    T = 150
    if sch_name == 'cos':
        sch = CosineAnnealingLR(T, lr_init, last_epoch=0, warmup=5)
    elif sch_name == 'multi_step':
        sch = MultiStepLR(lr_init, [50, 100], last_epoch=0, warmup=5)

    for step in range(150):
        lr = sch.step()
        if step % 20 == 0 or step < 20:
            print(f"[{step:3d}] lr={lr:.4f}")

    # resume
    print(f"Resume from step{step} with lr={lr:.4f}")
    T = 300
    if sch_name == 'cos':
        sch = CosineAnnealingLR(T, lr_init, last_epoch=step)
    elif sch_name == 'multi_step':
        sch = MultiStepLR(lr_init, [2, 4, 4, 50], last_epoch=step)
    for step in range(step, step+10):
        lr = sch.step()
        print(f"[{step:3d}] lr={lr:.4f}")


if __name__ == '__main__':
    test_lr_sch('cos')
