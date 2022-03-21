import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from utils.data_utils import DomainNetDataset, DigitsDataset, Partitioner, \
    CifarDataset, ClassWisePartitioner, extract_labels


def compose_transforms(trns, image_norm):
    if image_norm == '0.5':
        return transforms.Compose(trns + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif image_norm == 'torch':
        return transforms.Compose(trns + [transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))])
    elif image_norm == 'torch-resnet':
        return transforms.Compose(trns + [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    elif image_norm == 'none':
        return transforms.Compose(trns)
    else:
        raise ValueError(f"Invalid image_norm: {image_norm}")


def get_central_data(name: str, domains: list, percent=1., image_norm='none',
                     disable_image_norm_error=False):
    if image_norm != 'none' and not disable_image_norm_error:
        raise RuntimeError(f"This is a hard warning. Use image_norm != none will make the PGD"
                           f" attack invalid since PGD will clip the image into [0,1] range. "
                           f"Think before you choose {image_norm} image_norm.")
    if percent != 1. and name.lower() != 'digits':
        raise RuntimeError(f"percent={percent} should not be used in get_central_data."
                           f" Pass it to make_fed_data instead.")
    if name.lower() == 'digits':
        if image_norm == 'default':
            image_norm = '0.5'
        for domain in domains:
            if domain not in DigitsDataset.all_domains:
                raise ValueError(f"Invalid domain: {domain}")
        # Prepare data
        trns = {
            'MNIST': [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ],
            'SVHN': [
                transforms.Resize([28,28]),
                transforms.ToTensor(),
            ],
            'USPS': [
                transforms.Resize([28,28]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ],
            'SynthDigits': [
                transforms.Resize([28,28]),
                transforms.ToTensor(),
            ],
            'MNIST_M': [
                transforms.ToTensor(),
            ],
        }

        train_sets = [DigitsDataset(domain,
                                    percent=percent, train=True,
                                    transform=compose_transforms(trns[domain], image_norm))
                      for domain in domains]
        test_sets = [DigitsDataset(domain,
                                   train=False,
                                   transform=compose_transforms(trns[domain], image_norm))
                     for domain in domains]
    elif name.lower() in ('domainnet', 'domainnetf'):
        transform_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ])

        train_sets = [
            DomainNetDataset(domain, transform=transform_train,
                             full_set=name.lower()=='domainnetf')
            for domain in domains
        ]
        test_sets = [
            DomainNetDataset(domain, transform=transform_test, train=False,
                             full_set=name.lower()=='domainnetf')
            for domain in domains
        ]
    elif name.lower() == 'cifar10':
        if image_norm == 'default':
            image_norm = 'torch'
        for domain in domains:
            if domain not in CifarDataset.all_domains:
                raise ValueError(f"Invalid domain: {domain}")
        trn_train = [transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor()]
        trn_test = [transforms.ToTensor()]

        train_sets = [CifarDataset(domain, train=True,
                                   transform=compose_transforms(trn_train, image_norm))
                      for domain in domains]
        test_sets = [CifarDataset(domain, train=False,
                                  transform=compose_transforms(trn_test, image_norm))
                     for domain in domains]
    else:
        raise NotImplementedError(f"name: {name}")
    return train_sets, test_sets


def make_fed_data(train_sets, test_sets, batch_size, domains, shuffle_eval=False,
                  n_user_per_domain=1, partition_seed=42, partition_mode='uni',
                  n_class_per_user=-1, val_ratio=0.2,
                  eq_domain_train_size=True, percent=1.,
                  num_workers=0, pin_memory=False, min_n_sample_per_share=128,
                  subset_with_logits=False,
                  test_batch_size=None, shuffle=True,
                  consistent_test_class=False):
    """Distribute multi-domain datasets (`train_sets`) into federated clients.

    Args:
        train_sets (list): A list of datasets for training.
        test_sets (list): A list of datasets for testing.
        partition_seed (int): Seed for partitioning data into clients.
        consistent_test_class (bool): Ensure the test classes are the same training for a client.
            Meanwhile, make test sets are uniformly splitted for clients.
    """
    test_batch_size = batch_size if test_batch_size is None else test_batch_size
    SubsetClass = SubsetWithLogits if subset_with_logits else Subset
    clients = [f'{i}' for i in range(len(domains))]

    print(f" train size: {[len(s) for s in train_sets]}")
    print(f" test  size: {[len(s) for s in test_sets]}")

    train_len = [len(s) for s in train_sets]
    if eq_domain_train_size:
        train_len = [min(train_len)] * len(train_sets)
        # assert all([len(s) == train_len[0] for s in train_sets]), f"Should be equal length."

    if percent < 1:
        train_len = [int(tl * percent) for tl in train_len]

    print(f" trimmed train size: {[tl for tl in train_len]}")

    if n_user_per_domain > 1:  # split data into multiple users
        if n_class_per_user > 0:  # split by class-wise non-iid
            split = ClassWisePartitioner(rng=np.random.RandomState(partition_seed),
                                         n_class_per_share=n_class_per_user,
                                         min_n_sample_per_share=min_n_sample_per_share,
                                         partition_mode=partition_mode,
                                         verbose=True)
            splitted_clients = []
            val_sets, sub_train_sets, user_ids_by_class = [], [], []
            for i_client, (dname, tr_set) in enumerate(zip(clients, train_sets)):
                _tr_labels = extract_labels(tr_set)  # labels in the original order
                _tr_labels = _tr_labels[:train_len[i_client]]  # trim
                _idx_by_user, _user_ids_by_cls = split(_tr_labels, n_user_per_domain,
                                                       return_user_ids_by_class=True)
                print(f" {dname} | train split size: {[len(idxs) for idxs in _idx_by_user]}")
                _tr_labels = np.array(_tr_labels)
                print(f"    | train classes: "
                      f"{[f'{np.unique(_tr_labels[idxs]).tolist()}' for idxs in _idx_by_user]}")

                for i_user, idxs in zip(range(n_user_per_domain), _idx_by_user):
                    vl = int(val_ratio * len(idxs))

                    np.random.shuffle(idxs)
                    sub_train_sets.append(SubsetClass(tr_set, idxs[vl:]))

                    np.random.shuffle(idxs)
                    val_sets.append(Subset(tr_set, idxs[:vl]))

                    splitted_clients.append(f"{dname}-{i_user}")
                user_ids_by_class.append(_user_ids_by_cls if consistent_test_class else None)

            if consistent_test_class:
                # recreate partitioner to make sure consistent class distribution.
                split = ClassWisePartitioner(rng=np.random.RandomState(partition_seed),
                                             n_class_per_share=n_class_per_user,
                                             min_n_sample_per_share=min_n_sample_per_share,
                                             partition_mode='uni',
                                             verbose=True)
            sub_test_sets = []
            for i_client, te_set in enumerate(test_sets):
                _te_labels = extract_labels(te_set)
                _idx_by_user = split(_te_labels, n_user_per_domain,
                                     user_ids_by_class=user_ids_by_class[i_client])
                print(f"   test split size: {[len(idxs) for idxs in _idx_by_user]}")
                _te_labels = np.array(_te_labels)
                print(f"   test classes: "
                      f"{[f'{np.unique(_te_labels[idxs]).tolist()}' for idxs in _idx_by_user]}")

                for idxs in _idx_by_user:
                    np.random.shuffle(idxs)
                    sub_test_sets.append(Subset(te_set, idxs))
        else:  # class iid
            split = Partitioner(rng=np.random.RandomState(partition_seed),
                                min_n_sample_per_share=min_n_sample_per_share,
                                partition_mode=partition_mode)
            splitted_clients = []

            val_sets, sub_train_sets = [], []
            for i_client, (dname, tr_set) in enumerate(zip(clients, train_sets)):
                _train_len_by_user = split(train_len[i_client], n_user_per_domain)
                print(f" {dname} | train split size: {_train_len_by_user}")

                base_idx = 0
                for i_user, tl in zip(range(n_user_per_domain), _train_len_by_user):
                    vl = int(val_ratio * tl)
                    tl = tl - vl

                    sub_train_sets.append(SubsetClass(tr_set, list(range(base_idx, base_idx + tl))))
                    base_idx += tl

                    val_sets.append(Subset(tr_set, list(range(base_idx, base_idx + vl))))
                    base_idx += vl

                    splitted_clients.append(f"{dname}-{i_user}")

            # uniformly distribute test sets
            if consistent_test_class:
                split = Partitioner(rng=np.random.RandomState(partition_seed),
                                    min_n_sample_per_share=min_n_sample_per_share,
                                    partition_mode='uni')
            sub_test_sets = []
            for te_set in test_sets:
                _test_len_by_user = split(len(te_set), n_user_per_domain)

                base_idx = 0
                for tl in _test_len_by_user:
                    sub_test_sets.append(Subset(te_set, list(range(base_idx, base_idx + tl))))
                    base_idx += tl

        # rename
        train_sets = sub_train_sets
        test_sets = sub_test_sets
        clients = splitted_clients
    else:  # single user
        assert n_class_per_user <= 0, "Cannot split in Non-IID way when only one user for one " \
                                      f"domain. But got n_class_per_user={n_class_per_user}"
        val_len = [int(tl * val_ratio) for tl in train_len]

        val_sets = [Subset(tr_set, list(range(train_len[i_client]-val_len[i_client],
                                              train_len[i_client])))
                    for i_client, tr_set in enumerate(train_sets)]
        train_sets = [Subset(tr_set, list(range(train_len[i_client]-val_len[i_client])))
                      for i_client, tr_set in enumerate(train_sets)]

    # check the real sizes
    print(f" split users' train size: {[len(ts) for ts in train_sets]}")
    print(f" split users' val   size: {[len(ts) for ts in val_sets]}")
    print(f" split users' test  size: {[len(ts) for ts in test_sets]}")
    if val_ratio > 0:
        for i_ts, ts in enumerate(val_sets):
            if len(ts) <= 0:
                raise RuntimeError(f"user-{i_ts} not has enough val data.")

    train_loaders = [DataLoader(tr_set, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=pin_memory,
                                drop_last=partition_mode != 'uni') for tr_set in train_sets]
    test_loaders = [DataLoader(te_set, batch_size=test_batch_size, shuffle=shuffle_eval,
                               num_workers=num_workers, pin_memory=pin_memory)
                    for te_set in test_sets]
    if val_ratio > 0:
        val_loaders = [DataLoader(va_set, batch_size=batch_size, shuffle=shuffle_eval,
                                  num_workers=num_workers, pin_memory=pin_memory)
                       for va_set in val_sets]
    else:
        val_loaders = test_loaders

    return train_loaders, val_loaders, test_loaders, clients

def prepare_domainnet_data(args, domains=['clipart', 'quickdraw'], shuffle_eval=False,
                           n_class_per_user=-1, n_user_per_domain=1,
                           partition_seed=42, partition_mode='uni',
                           val_ratio=0., eq_domain_train_size=True,
                           subset_with_logits=False, consistent_test_class=False,
                           ):
    assert args.data.lower() in ['domainnet', 'domainnetf']
    train_sets, test_sets = get_central_data(args.data.lower(), domains)

    train_loaders, val_loaders, test_loaders, clients = make_fed_data(
        train_sets, test_sets, args.batch, domains, shuffle_eval=shuffle_eval,
        partition_seed=partition_seed, n_user_per_domain=n_user_per_domain,
        partition_mode=partition_mode,
        val_ratio=val_ratio, eq_domain_train_size=eq_domain_train_size, percent=args.percent,
        min_n_sample_per_share=16, subset_with_logits=subset_with_logits,
        n_class_per_user=n_class_per_user,
        test_batch_size=args.test_batch if hasattr(args, 'test_batch') else args.batch,
        num_workers=8 if args.data.lower() == 'domainnetf' else 0,
        pin_memory=False if args.data.lower() == 'domainnetf' else True,
        consistent_test_class=consistent_test_class,
    )
    return train_loaders, val_loaders, test_loaders, clients


def prepare_digits_data(args, domains=['MNIST', 'SVHN'], shuffle_eval=False, n_class_per_user=-1,
                        n_user_per_domain=1, partition_seed=42, partition_mode='uni', val_ratio=0.2,
                        eq_domain_train_size=True, subset_with_logits=False,
                        consistent_test_class=False,
                        ):
    do_adv_train = hasattr(args, 'noise') and (args.noise == 'none' or args.noise_ratio == 0
                                               or args.n_noise_domain == 0)
    # NOTE we use the image_norm=0.5 for reproducing clean training results.
    #   but for adv training, we do not use image_norm
    train_sets, test_sets = get_central_data(
        args.data, domains, percent=args.percent, image_norm='0.5' if do_adv_train else 'none',
        disable_image_norm_error=True)
    train_loaders, val_loaders, test_loaders, clients = make_fed_data(
        train_sets, test_sets, args.batch, domains, shuffle_eval=shuffle_eval,
        partition_seed=partition_seed, n_user_per_domain=n_user_per_domain,
        partition_mode=partition_mode,
        val_ratio=val_ratio, eq_domain_train_size=eq_domain_train_size,
        min_n_sample_per_share=16, n_class_per_user=n_class_per_user,
        subset_with_logits=subset_with_logits,
        test_batch_size=args.test_batch if hasattr(args, 'test_batch') else args.batch,
        consistent_test_class=consistent_test_class,
    )
    return train_loaders, val_loaders, test_loaders, clients


def prepare_cifar_data(args, domains=['cifar10'], shuffle_eval=False, n_class_per_user=-1,
                       n_user_per_domain=1, partition_seed=42, partition_mode='uni', val_ratio=0.2,
                       eq_domain_train_size=True, subset_with_logits=False,
                       consistent_test_class=False,
                       ):
    train_sets, test_sets = get_central_data('cifar10', domains)

    train_loaders, val_loaders, test_loaders, clients = make_fed_data(
        train_sets, test_sets, args.batch, domains, shuffle_eval=shuffle_eval,
        partition_seed=partition_seed, n_user_per_domain=n_user_per_domain,
        partition_mode=partition_mode,
        val_ratio=val_ratio, eq_domain_train_size=eq_domain_train_size, percent=args.percent,
        min_n_sample_per_share=64 if n_class_per_user > 3 else 16, subset_with_logits=subset_with_logits,
        n_class_per_user=n_class_per_user,
        test_batch_size=args.test_batch if hasattr(args, 'test_batch') else args.batch,
        consistent_test_class=consistent_test_class,
    )
    return train_loaders, val_loaders, test_loaders, clients


class SubsetWithLogits(Subset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices) -> None:
        super(SubsetWithLogits, self).__init__(dataset, indices)
        self.logits = [0. for _ in range(len(indices))]

    def __getitem__(self, idx):
        dataset_subset = self.dataset[self.indices[idx]]
        if isinstance(dataset_subset, tuple):
            return (*dataset_subset, self.logits[idx])
        else:
            return dataset_subset, self.logits[idx]

    def update_logits(self, idx, logit):
        self.logits[idx] = logit


if __name__ == '__main__':
    data = 'cifar10'
    if data == 'digits':
        train_loaders, val_loaders, test_loaders, clients = prepare_digits_data(
            type('MockClass', (object,), {'percent': 1.0, 'batch': 32}), domains=['MNIST'],
            n_user_per_domain=5,
            partition_seed=1,
            partition_mode='uni',
            val_ratio=0.2,
        )
        for batch in train_loaders[0]:
            data, target = batch
            print(target)
            break
    elif data == 'cifar10':
        train_loaders, val_loaders, test_loaders, clients = prepare_cifar_data(
            type('MockClass', (object,), {'batch': 32, 'percent': 0.1}), domains=['cifar10'],
            n_user_per_domain=5,
            partition_seed=1,
            partition_mode='uni',
            val_ratio=0.2,
            subset_with_logits=True
        )
        for batch in train_loaders[0]:
            smp_idxs, data, target, t_logits = batch
            print(smp_idxs)
            break

        temp_loader = DataLoader(train_loaders[0].dataset, batch_size=32, shuffle=False)
        all_logits = []
        for batch in temp_loader:
            # FIXME need to modify SubsetWithLogits to return the index
            smp_idxs, data, target, t_logits = batch
            all_logits.append(torch.rand((len(data), 10)))
            # for i in smp_idxs
            # print(smp_idxs)
        all_logits = torch.cat(all_logits, dim=0)
        assert isinstance(train_loaders[0].dataset, SubsetWithLogits)
        train_loaders[0].dataset.logits = all_logits

        for batch in train_loaders[0]:
            smp_idxs, data, target, t_logits = batch
            print("t_logits shape", t_logits.shape)
            break
