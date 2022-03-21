"""Core functions of federate learning."""
import argparse
import copy

import numpy as np
from advertorch.attacks import LinfPGDAttack
from torch import nn

from federated.aggregation import ModelAccumulator, SlimmableModelAccumulator
from nets.slimmable_models import get_slim_ratios_from_str, parse_lognorm_slim_schedule
from utils.utils import shuffle_sampler, str2bool


class _Federation:
    """A helper class for federated data creation.
    Use `add_argument` to setup ArgumentParser and then use parsed args to init the class.
    """
    _model_accum: ModelAccumulator

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser):
        # data
        parser.add_argument('--percent', type=float, default=0.3,
                            help='percentage of dataset for training')
        parser.add_argument('--val_ratio', type=float, default=0.5,
                            help='ratio of train set for validation')
        parser.add_argument('--batch', type=int, default=32, help='batch size')
        parser.add_argument('--test_batch', type=int, default=128, help='batch size for test')

        # federated
        parser.add_argument('--pd_nuser', type=int, default=10, help='#users per domain.')
        parser.add_argument('--pr_nuser', type=int, default=-1, help='#users per comm round '
                                                                     '[default: all]')
        parser.add_argument('--pu_nclass', type=int, default=-1, help='#class per user. -1 or 0: all')
        parser.add_argument('--domain_order', choices=list(range(5)), type=int, default=0,
                            help='select the order of domains')
        parser.add_argument('--partition_mode', choices=['uni', 'dir'], type=str.lower, default='uni',
                            help='the mode when splitting domain data into users: uni - uniform '
                                 'distribution (all user have the same #samples); dir - Dirichlet'
                                 ' distribution (non-iid sample sizes)')
        parser.add_argument('--con_test_cls', action='store_true',
                            help='Ensure the test classes are the same training for a client. '
                                 'Meanwhile, make test sets are uniformly splitted for clients. '
                                 'Mainly influence class-niid settings.')

    @classmethod
    def render_run_name(cls, args):
        run_name = f'__pd_nuser_{args.pd_nuser}'
        if args.percent != 0.3: run_name += f'__pct_{args.percent}'
        if args.pu_nclass > 0: run_name += f"__pu_nclass_{args.pu_nclass}"
        if args.pr_nuser != -1: run_name += f'__pr_nuser_{args.pr_nuser}'
        if args.domain_order != 0: run_name += f'__do_{args.domain_order}'
        if args.partition_mode != 'uni': run_name += f'__part_md_{args.partition_mode}'
        if args.con_test_cls: run_name += '__ctc'
        return run_name

    def __init__(self, data, args):
        self.args = args

        # Prepare Data
        num_classes = 10
        if data == 'Digits':
            from utils.data_utils import DigitsDataset
            from utils.data_loader import prepare_digits_data
            prepare_data = prepare_digits_data
            DataClass = DigitsDataset
        elif data == 'DomainNet':
            from utils.data_utils import DomainNetDataset
            from utils.data_loader import prepare_domainnet_data
            prepare_data = prepare_domainnet_data
            DataClass = DomainNetDataset
        elif data == 'Cifar10':
            from utils.data_utils import CifarDataset
            from utils.data_loader import prepare_cifar_data
            prepare_data = prepare_cifar_data
            DataClass = CifarDataset
        else:
            raise ValueError(f"Unknown dataset: {data}")
        all_domains = DataClass.resorted_domains[args.domain_order]

        train_loaders, val_loaders, test_loaders, clients = prepare_data(
            args, domains=all_domains,
            n_user_per_domain=args.pd_nuser,
            n_class_per_user=args.pu_nclass,
            partition_seed=args.seed + 1,
            partition_mode=args.partition_mode,
            val_ratio=args.val_ratio,
            eq_domain_train_size=args.partition_mode == 'uni',
            consistent_test_class=args.con_test_cls,
        )
        clients = [c + ' ' + ('noised' if hasattr(args, 'adv_lmbd') and args.adv_lmbd > 0.
                              else 'clean') for c in clients]

        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders
        self.clients = clients
        self.num_classes = num_classes
        self.all_domains = all_domains

        # Setup fed
        self.client_num = len(self.clients)
        client_weights = [len(tl.dataset) for tl in train_loaders]
        self.client_weights = [w / sum(client_weights) for w in client_weights]

        pr_nuser = args.pr_nuser if args.pr_nuser > 0 else self.client_num
        self.args.pr_nuser = pr_nuser
        self.client_sampler = UserSampler([i for i in range(self.client_num)], pr_nuser, mode='uni')

    def get_data(self):
        return self.train_loaders, self.val_loaders, self.test_loaders

    def make_aggregator(self, running_model):
        self._model_accum = ModelAccumulator(running_model, self.args.pr_nuser, self.client_num)
        return self._model_accum

    @property
    def model_accum(self):
        if not hasattr(self, '_model_accum'):
            raise RuntimeError(f"model_accum has not been set yet. Call `make_aggregator` first.")
        return self._model_accum

    def download(self, running_model, client_idx, strict=True):
        """Download (personalized) global model to running_model."""
        self.model_accum.load_model(running_model, client_idx, strict=strict)

    def upload(self, running_model, client_idx):
        """Upload client model."""
        self.model_accum.add(client_idx, running_model, self.client_weights[client_idx])

    def aggregate(self):
        """Aggregate received models and update global model."""
        self.model_accum.update_server_and_reset()


class HeteFederation(_Federation):
    """Heterogeneous federation where each client is capable for training different widths."""
    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser):
        super(HeteFederation, cls).add_argument(parser)
        parser.add_argument('--slim_ratios', type=str, default='8-4-2-1',
                            help='define the slim_ratio for groups, for example, 8-4-2-1 [default]'
                                 ' means x1/8 net for the 1st group, and x1/4 for the 2nd')
        parser.add_argument('--val_ens_only', action='store_true',
                            help='only validate the full-size model')

    @classmethod
    def render_run_name(cls, args):
        run_name = super(HeteFederation, cls).render_run_name(args)
        if args.slim_ratios != '8-4-2-1': run_name += f'__{args.slim_ratios}'
        return run_name

    def __init__(self, data, args):
        super(HeteFederation, self).__init__(data, args)
        train_slim_ratios = get_slim_ratios_from_str(args.slim_ratios)
        if len(train_slim_ratios) <= 1:
            info = f"WARN: There is no width to customize for training with " \
                  f"slim_ratios={args.slim_ratios}. To set a non-single" \
                  f" slim_ratios."
            if len(train_slim_ratios) > 0:
                print(info)
            else:
                raise RuntimeError(info)
        max_slim_ratio = max(train_slim_ratios)
        if args.val_ens_only:
            val_slim_ratios = [max_slim_ratio]  # only validate the max width
        else:
            val_slim_ratios = copy.deepcopy(train_slim_ratios)
            if max_slim_ratio not in val_slim_ratios:
                val_slim_ratios.append(max_slim_ratio)  # make sure the max width model is validated.

        self.train_slim_ratios = train_slim_ratios
        self.user_max_slim_ratios = self.get_slim_ratio_schedule(train_slim_ratios, args.slim_ratios)
        self.val_slim_ratios = val_slim_ratios

    def get_slim_ratio_schedule(self, train_slim_ratios: list, mode: str):
        if mode.startswith('ln'):  # lognorm
            return parse_lognorm_slim_schedule(train_slim_ratios, mode, self.client_num)
        else:
            return [train_slim_ratios[int(len(train_slim_ratios) * i / self.client_num)]
                    for i, cname in enumerate(self.clients)]

    def make_aggregator(self, running_model, local_bn=False):
        self._model_accum = SlimmableModelAccumulator(running_model, self.args.pr_nuser,
                                                      self.client_num, local_bn=local_bn)
        return self._model_accum

    def upload(self, running_model, client_idx, max_slim_ratio=None, slim_bias_idx=None):
        assert max_slim_ratio is not None
        assert slim_bias_idx is not None
        self.model_accum.add(client_idx, running_model, self.client_weights[client_idx],
                             max_slim_ratio=max_slim_ratio, slim_bias_idx=slim_bias_idx)

    def sample_bases(self, client_idx):
        """Sample slimmer base models for the client.
        Return slim_ratios, slim_shifts
        """
        max_slim_ratio = self.user_max_slim_ratios[client_idx]
        slim_shifts = [0]
        slim_ratios = [max_slim_ratio]
        print(f" max slim ratio: {max_slim_ratio} "
              f"slim_ratios={slim_ratios}, slim_shifts={slim_shifts}")
        return slim_ratios, slim_shifts


class SHeteFederation(HeteFederation):
    """Extend HeteroFL w/ local slimmable training."""
    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser):
        super(SHeteFederation, cls).add_argument(parser)
        parser.add_argument('--slimmable_train', type=str2bool, default=True,
                            help='train all budget-compatible slimmable networks, otherwise HeteroFL')

    @classmethod
    def render_run_name(cls, args):
        run_name = super(SHeteFederation, cls).render_run_name(args)
        if not args.slimmable_train: run_name += f'__nst'
        return run_name

    def sample_bases(self, client_idx):
        """Sample slimmer base models for the client.
        Return slim_ratios, slim_shifts
        """
        max_slim_ratio = self.user_max_slim_ratios[client_idx]
        if self.args.slimmable_train:
            if len(self.train_slim_ratios) > 4:
                print("WARN: over 4 trained slim ratios which will cause large overhead for"
                      " slimmable training. Try to set slimmable_train=False (HeteroFL) instead.")
            slim_ratios = [r for r in self.train_slim_ratios if r <= max_slim_ratio]
        else:
            slim_ratios = [max_slim_ratio]
        slim_shifts = [0] * len(slim_ratios)
        print(f" max slim ratio: {max_slim_ratio} "
              f"slim_ratios={slim_ratios}, slim_shifts={slim_shifts}")
        return slim_ratios, slim_shifts


class SplitFederation(HeteFederation):
    """Split a net into multiple subnets and train them in federated learning."""
    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser):
        super(SplitFederation, cls).add_argument(parser)
        parser.add_argument('--atom_slim_ratio', type=float, default=0.125,
                            help='the width ratio of a base model')

    @classmethod
    def render_run_name(cls, args):
        run_name = super(SplitFederation, cls).render_run_name(args)
        assert 0. < args.atom_slim_ratio <= 1., f"Invalid slim_ratio: {args.atom_slim_ratio}"
        if args.atom_slim_ratio != 0.125: run_name += f"__asr{args.atom_slim_ratio}"
        return run_name

    def __init__(self, data, args):
        super(SplitFederation, self).__init__(data, args)

        assert args.atom_slim_ratio <= min(self.train_slim_ratios), \
            f"Base model's width ({args.atom_slim_ratio}) is larger than that of minimal allowed " \
            f"width ({min(self.train_slim_ratios)})"

        self.num_base = int(max(self.train_slim_ratios) / args.atom_slim_ratio)
        self.user_base_sampler = shuffle_sampler(list(range(self.num_base)))

    def sample_bases(self, client_idx):
        """Sample base models for the client.
        Return slim_ratios, slim_shifts
        """
        # (Alg 2) Sample base models defined by shift index.
        max_slim_ratio = self.user_max_slim_ratios[client_idx]
        user_n_base = int(max_slim_ratio / self.args.atom_slim_ratio)

        slim_shifts = [self.user_base_sampler.next()]
        if user_n_base > 1:
            _sampler = shuffle_sampler([v for v in self.user_base_sampler.arr if v != slim_shifts[0]])
            slim_shifts += [_sampler.next() for _ in range(user_n_base - 1)]
        slim_ratios = [self.args.atom_slim_ratio] * user_n_base
        print(f" max slim ratio: {max_slim_ratio} "
              f"slim_ratios={slim_ratios}, slim_shifts={slim_shifts}")
        return slim_ratios, slim_shifts


class UserSampler(object):
    def __init__(self, users, select_nuser, mode='all'):
        self.users = users
        self.total_num_user = len(users)
        self.select_nuser = select_nuser
        self.mode = mode
        if mode == 'all':
            assert select_nuser == self.total_num_user, "Conflict config: Select too few users."

    def iter(self):
        if self.mode == 'all' or self.select_nuser == self.total_num_user:
            sel = np.arange(len(self.users))
        elif self.mode == 'uni':
            sel = np.random.choice(self.total_num_user, self.select_nuser, replace=False)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        for i in sel:
            yield self.users[i]


class AdversaryCreator(object):
    """A factory producing adversary.

    Args:
        attack: Name. MIA for MomentumIterativeAttack with Linf norm. LSA for LocalSearchAttack.
        eps: Constraint on the distortion norm
        steps: Number of attack steps
    """
    supported_adv = ['LinfPGD', 'LinfPGD20', 'LinfPGD20_eps16', 'LinfPGD100','LinfPGD100_eps16',
                     'LinfPGD4_eps4', 'LinfPGD3_eps4', 'LinfPGD7_eps4',
                     ]

    def __init__(self, attack: str, **kwargs):
        self.attack = attack
        if '_eps' in self.attack:
            self.attack, default_eps = self.attack.split('_eps')
            self.eps = kwargs.setdefault('eps', int(default_eps))
        else:
            self.eps = kwargs.setdefault('eps', 8.)
        if self.attack.startswith('LinfPGD') and self.attack[len('LinfPGD'):].isdigit():
            assert 'steps' not in kwargs, "The steps is set by the attack name while " \
                                          "found additional set in kwargs."
            self.steps = int(self.attack[len('LinfPGD'):])
        elif self.attack.startswith('MIA') and self.attack[len('MIA'):].isdigit():
            assert 'steps' not in kwargs, "The steps is set by the attack name while " \
                                          "found additional set in kwargs."
            self.steps = int(self.attack[len('MIA'):])
        else:
            self.steps = kwargs.setdefault('steps', 7)

    def __call__(self, model):
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        if self.attack.startswith('LinfPGD'):
            adv = LinfPGDAttack(
                model, loss_fn=loss_fn, eps=self.eps / 255,
                nb_iter=self.steps,
                eps_iter=min(self.eps / 255 * 1.25, self.eps / 255 + 4. / 255) / self.steps,
                rand_init=True,
                clip_min=0.0, clip_max=1.0,
                targeted=False)
        elif self.attack == 'none':
            adv = None
        else:
            raise ValueError(f"attack: {self.attack}")
        return adv
