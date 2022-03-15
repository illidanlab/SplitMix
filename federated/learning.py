import sys

import numpy as np
import torch
from advertorch.context import ctx_noparamgrad_and_eval
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from federated.core import AdversaryCreator
from utils.utils import AverageMeter
from nets.dual_bn import set_bn_mode


def if_use_dbn(model):
    if isinstance(model, DDP):
        return model.module.bn_type.startswith('d')
    else:
        return model.bn_type.startswith('d')


def train(model, data_loader, optimizer, loss_fun, device, adversary=None, adv_lmbd=0.5,
          start_iter=0, max_iter=np.inf, att_BNn=False, progress=True):

    model.train()
    loss_all = 0
    total = 0
    correct = 0
    max_iter = len(data_loader) if max_iter == np.inf else max_iter
    data_iterator = iter(data_loader)
    tqdm_iters = tqdm(range(start_iter, max_iter), file=sys.stdout) \
        if progress else range(start_iter, max_iter)
    if adversary is None:
        # ordinary training.
        set_bn_mode(model, False)  # set clean mode
        for step in tqdm_iters:
        # for data, target in tqdm(data_loader, file=sys.stdout):
            try:
                data, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                data, target = next(data_iterator)
            optimizer.zero_grad()

            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fun(output, target)

            loss_all += loss.item() * target.size(0)
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

            loss.backward()
            optimizer.step()
    else:
        # Use adversary to perturb data.
        for step in tqdm_iters:
            all_logits = []
            all_targets = []
            try:
                data, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                data, target = next(data_iterator)
            optimizer.zero_grad()

            # clean data
            data = data.to(device)
            target = target.to(device)
            if adv_lmbd < 1. or if_use_dbn(model):  # FIXME if dbn, the skip will make a void BNc
                set_bn_mode(model, False)  # set clean mode
                logits = model(data)
                clf_loss_clean = loss_fun(logits, target)
                all_logits.append(logits)
                all_targets.append(target)
            else:
                clf_loss_clean = 0.

            if adv_lmbd > 0 or if_use_dbn(model):
                # noise data
                # ---- use adv ----
                if att_BNn:
                    set_bn_mode(model, True)  # set noise mode
                with ctx_noparamgrad_and_eval(model):
                    noise_data = adversary.perturb(data, target)
                noise_target = target
                # -----------------

                set_bn_mode(model, True)  # set noise mode
                logits_noise = model(noise_data)
                clf_loss_noise = loss_fun(logits_noise, noise_target)
                all_logits.append(logits_noise)
                all_targets.append(noise_target)
            else:
                clf_loss_noise = 0

            loss = (1 - adv_lmbd) * clf_loss_clean + adv_lmbd * clf_loss_noise

            output = torch.cat(all_logits, dim=0)
            target = torch.cat(all_targets, dim=0)
            loss_all += loss.item() * target.size(0)
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

            loss.backward()
            optimizer.step()
    return loss_all / total, correct / total


def train_slimmable(model, data_loader, optimizer, loss_fun, device, adversary=None, adv_lmbd=0.5,
                    start_iter=0, max_iter=np.inf, att_BNn=False,
                    slim_ratios=[0.5, 0.75, 1.0], slim_shifts=0, out_slim_shifts=None,
                    progress=True, loss_temp='none'):
    """If slim_ratios is a single value, use `train` and set slim_ratio outside, instead.
    """
    # expand scalar slim_shift to list
    if not isinstance(slim_shifts, (list, tuple)):
        slim_shifts = [slim_shifts for _ in range(len(slim_ratios))]
    if not isinstance(out_slim_shifts, (list, tuple)):
        out_slim_shifts = [out_slim_shifts for _ in range(len(slim_ratios))]

    model.train()
    total, correct, loss_all = 0, 0, 0
    max_iter = len(data_loader) if max_iter == np.inf else max_iter
    data_iterator = iter(data_loader)
    if adversary is None:
        # ordinary training.
        set_bn_mode(model, False)  # set clean mode
        for step in tqdm(range(start_iter, max_iter), file=sys.stdout, disable=not progress):
            # for data, target in tqdm(data_loader, file=sys.stdout):
            try:
                data, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                data, target = next(data_iterator)
            optimizer.zero_grad()

            data = data.to(device)
            target = target.to(device)

            loss = 0.
            for slim_ratio, in_slim_shift, out_slim_shift \
                    in sorted(zip(slim_ratios, slim_shifts, out_slim_shifts), reverse=False,
                              key=lambda ss_pair: ss_pair[0]):
                model.switch_slim_mode(slim_ratio, slim_bias_idx=in_slim_shift, out_slim_bias_idx=out_slim_shift)

                output = model(data)
                if loss_temp == 'none':
                    _loss = loss_fun(output, target)
                elif loss_temp == 'auto':
                    _loss = loss_fun(output/slim_ratio, target) * slim_ratio
                elif loss_temp.replace('.', '', 1).isdigit():  # is float
                    _temp = float(loss_temp)
                    _loss = loss_fun(output / _temp, target) * _temp
                else:
                    raise NotImplementedError(f"loss_temp: {loss_temp}")

                loss_all += _loss.item() * target.size(0)
                total += target.size(0)
                pred = output.data.max(1)[1]
                correct += pred.eq(target.view(-1)).sum().item()

                _loss.backward()
            optimizer.step()
    else:
        # Use adversary to perturb data.
        for step in tqdm(range(start_iter, max_iter), file=sys.stdout, disable=not progress):
            # for data, target in tqdm(data_loader, file=sys.stdout):
            try:
                data, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                data, target = next(data_iterator)
            optimizer.zero_grad()

            # clean data
            data = data.to(device)
            target = target.to(device)

            for slim_ratio, in_slim_shift, out_slim_shift \
                    in sorted(zip(slim_ratios, slim_shifts, out_slim_shifts), reverse=False,
                              key=lambda ss_pair: ss_pair[0]):
                # TODO also set mode at test.
                model.switch_slim_mode(slim_ratio, slim_bias_idx=in_slim_shift,
                                       out_slim_bias_idx=out_slim_shift)

                all_logits, all_targets = [], []

                if adv_lmbd < 1. or if_use_dbn(model):  # FIXME if dbn, the skip will make a void BNc
                    set_bn_mode(model, False)  # set clean mode
                    logits = model(data)
                    clf_loss_clean = loss_fun(logits, target)
                    all_logits.append(logits)
                    all_targets.append(target)
                else:
                    clf_loss_clean = 0.

                if adv_lmbd > 0 or if_use_dbn(model):
                    # noise data
                    # ---- use adv ----
                    if att_BNn:
                        set_bn_mode(model, True)  # set noise mode
                    with ctx_noparamgrad_and_eval(model):
                        noise_data = adversary.perturb(data, target)
                    noise_target = target
                    # -----------------

                    set_bn_mode(model, True)  # set noise mode
                    logits_noise = model(noise_data)
                    clf_loss_noise = loss_fun(logits_noise, noise_target)
                    all_logits.append(logits_noise)
                    all_targets.append(noise_target)
                else:
                    clf_loss_noise = 0

                loss = (1 - adv_lmbd) * clf_loss_clean + adv_lmbd * clf_loss_noise

                loss_all += loss.item() * target.size(0)

                all_logits_t = torch.cat(all_logits, dim=0)
                all_targets_t = torch.cat(all_targets, dim=0)
                total += all_targets_t.size(0)
                pred = all_logits_t.data.max(1)[1]
                correct += pred.eq(all_targets_t.view(-1)).sum().item()

                loss.backward()
            optimizer.step()
    return loss_all / total, correct / total


# =========== Test ===========


def test_dbn(model, data_loader, loss_fun, device,
             adversary=None, detector=None, att_BNn=False, adversary_name=None, progress=False,
             mix_dual_logit_lmbd=-1, attack_mix_dual_logit_lmbd=-1, deep_mix=False,
             ):
    model.eval()

    noise_type = 1 if adversary else 0
    loss_all = 0
    total = 0
    correct = 0
    tqdm_data_loader = tqdm(data_loader, file=sys.stdout) if progress else data_loader
    for data, target in tqdm_data_loader:
        data = data.to(device)
        target = target.to(device)
        set_bn_mode(model, is_noised=False)  # use clean mode to predict noise.
        if adversary:
            if mix_dual_logit_lmbd >= 0:
                if attack_mix_dual_logit_lmbd < 0:
                    attack_mix_dual_logit_lmbd = mix_dual_logit_lmbd
                joint_adversary = AdversaryCreator(adversary_name)(
                    lambda x: model.mix_dual_forward(x, lmbd=attack_mix_dual_logit_lmbd,
                                                     deep_mix=deep_mix)
                )
                with ctx_noparamgrad_and_eval(model):  # make sure BN's are in eval mode
                    data = joint_adversary.perturb(data, target)
            else:
                set_bn_mode(model, att_BNn)  # set noise mode
                with ctx_noparamgrad_and_eval(model):  # make sure BN's are in eval mode
                    data = adversary.perturb(data, target)

        if detector is None or detector == 'none':
            # use clean BN
            set_bn_mode(model, is_noised=False)
        elif isinstance(detector, str):
            if detector == 'clean':
                disc_pred = False
            elif detector == 'noised':
                disc_pred = True
            elif detector == 'gt':
                disc_pred = noise_type > 0
            elif detector == 'rgt':
                disc_pred = noise_type <= 0
            else:
                raise ValueError(f"Invalid str detector: {detector}")
            set_bn_mode(model, is_noised=disc_pred)
        else:
            raise NotImplementedError("Not support detector model.")

        if mix_dual_logit_lmbd >= 0:
            output = model.mix_dual_forward(data, lmbd=mix_dual_logit_lmbd, deep_mix=deep_mix)
        else:
            output = model(data)
        loss = loss_fun(output, target)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct / total


def test(model, data_loader, loss_fun, device, adversary=None, progress=False):
    """Run test single model.
    Returns: loss, acc
    """
    model.eval()
    loss_all, total, correct = 0, 0, 0
    for data, target in tqdm(data_loader, file=sys.stdout, disable=not progress):
        data, target = data.to(device), target.to(device)
        if adversary:
            with ctx_noparamgrad_and_eval(model):  # make sure BN's are in eval mode
                data = adversary.perturb(data, target)

        with torch.no_grad():
            output = model(data)
            loss = loss_fun(output, target)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def refresh_bn(model, data_loader, device, adversary=None, progress=False):
    model.train()
    for data, target in tqdm(data_loader, file=sys.stdout, disable=not progress):
        data, target = data.to(device), target.to(device)
        if adversary:
            with ctx_noparamgrad_and_eval(model):  # make sure BN's are in eval mode
                data = adversary.perturb(data, target)

        with torch.no_grad():
            model(data)


def fed_test_model(fed, running_model, test_loaders, loss_fun, device):
    test_acc_mt = AverageMeter()
    for test_idx, test_loader in enumerate(test_loaders):
        fed.download(running_model, test_idx)
        _, test_acc = test(running_model, test_loader, loss_fun, device)
        # print(' {:<11s}| Test  Acc: {:.4f}'.format(fed.clients[test_idx], test_acc))

        # wandb.summary[f'{fed.clients[test_idx]} test acc'] = test_acc
        test_acc_mt.append(test_acc)
    return test_acc_mt.avg

