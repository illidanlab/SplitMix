import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
import os
from tqdm import tqdm
from torchvision.datasets.folder import IMG_EXTENSIONS, has_file_allowed_extension

from typing import Tuple, List, Dict, Optional, Callable, cast
from .config import DATA_PATHS
from .utils import shuffle_sampler


class DigitsDataset(Dataset):
    all_domains = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST_M']
    resorted_domains = {
        0: ['MNIST',    'SVHN', 'USPS', 'SynthDigits', 'MNIST_M'],
        1: ['SVHN',     'USPS', 'SynthDigits', 'MNIST_M', 'MNIST'],
        2: ['USPS',     'SynthDigits', 'MNIST_M', 'MNIST', 'SVHN'],
        3: ['SynthDigits', 'MNIST_M', 'MNIST', 'SVHN', 'USPS'],
        4: ['MNIST_M',  'MNIST', 'SVHN', 'USPS', 'SynthDigits'],
    }
    num_classes = 10  # may not be correct

    def __init__(self, domain, percent=0.1, filename=None, train=True, transform=None):
        data_path = os.path.join(DATA_PATHS["Digits"], domain)
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(
                                os.path.join(data_path,
                                             'partitions/train_part{}.pkl'.format(part)),
                                allow_pickle=True)
                        else:
                            images, labels = np.load(
                                os.path.join(data_path,
                                             'partitions/train_part{}.pkl'.format(part)),
                                allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(
                        os.path.join(data_path, 'partitions/train_part0.pkl'),
                        allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'),
                                                   allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename),
                                               allow_pickle=True)

        self.transform = transform
        self.channels = 3 if domain in ['SVHN', 'SynthDigits', 'MNIST_M'] else 1
        self.labels = self.labels.astype(np.long).squeeze()
        self.classes = np.unique(self.labels)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CifarDataset(CIFAR10):
    all_domains = ['cifar10']
    resorted_domains = {
        0: ['cifar10'],
    }
    num_classes = 10  # may not be correct

    def __init__(self, domain='cifar10', train=True, transform=None, download=False):
        assert domain in self.all_domains, f"Invalid domain: {domain}"
        data_path = os.path.join(DATA_PATHS["Cifar10"], domain)
        super().__init__(data_path, train=train, transform=transform, download=download)


class DomainNetDataset(Dataset):
    all_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    resorted_domains = {
        0: ['real',      'clipart',   'infograph', 'painting',  'quickdraw', 'sketch'],
        1: ['clipart',   'infograph', 'painting',  'quickdraw', 'sketch',    'real'],
        2: ['infograph', 'painting',  'quickdraw', 'sketch',    'real',      'clipart'],
        3: ['painting',  'quickdraw', 'sketch',    'real',      'clipart',   'infograph'],
        4: ['quickdraw', 'sketch',    'real',      'clipart',   'infograph', 'painting'],
        5: ['sketch',    'real',      'clipart',   'infograph', 'painting',  'quickdraw'],
    }
    num_classes = 10  # may not be correct

    def __init__(self, site, train=True, transform=None, full_set=False):
        self.full_set = full_set
        self.base_path = DATA_PATHS['DomainNet']
        if full_set:
            classes, class_to_idx = find_classes(f"{self.base_path}/{site}")
            self.text_labels = classes
            self.paths, self.labels = make_dataset_from_dir(f"{self.base_path}/{site}",
                                                            class_to_idx, IMG_EXTENSIONS)
            self.num_classes = len(class_to_idx)
        else:
            self.paths, self.text_labels = np.load('{}/DomainNet/{}_{}.pkl'.format(
                DATA_PATHS['DomainNetPathList'],
                site, 'train' if train else 'test'), allow_pickle=True)

            class_to_idx = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4,
                            'tiger': 5, 'whale': 6, 'windmill': 7, 'wine_glass': 8, 'zebra': 9}

            self.labels = [class_to_idx[text] for text in self.text_labels]
            self.num_classes = len(class_to_idx)

        self.transform = transform
        self.classes = np.unique(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        site, cls, fname = self.paths[idx].split('/')[-3:]
        img_path = os.path.join(self.base_path, site, cls, fname)

        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# //////////// Data processing ////////////
def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx
            is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset_from_dir(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> Tuple[List[str], List[int]]:
    """Different Pytorch version, we return path and labels in two lists."""
    paths, labels = [], []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the "
                         "same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    paths.append(path)
                    labels.append(class_index)
                    # instances.append(item)
    return paths, labels


class Partitioner(object):
    """Partition a sequence into shares."""
    def __init__(self, rng=None, partition_mode="dir",
                 max_n_sample_per_share=-1,
                 min_n_sample_per_share=2,
                 max_n_sample=-1,
                 verbose=True
                 ):
        assert max_n_sample_per_share < 0 or max_n_sample_per_share > min_n_sample_per_share, \
            f"max ({max_n_sample_per_share}) > min ({min_n_sample_per_share})"
        self.rng = rng if rng else np.random
        self.partition_mode = partition_mode
        self.max_n_sample_per_share = max_n_sample_per_share
        self.min_n_sample_per_share = min_n_sample_per_share
        self.max_n_sample = max_n_sample
        self.verbose = verbose

    def __call__(self, n_sample, n_share, log=print):
        """Partition a sequence of `n_sample` into `n_share` shares.
        Returns:
            partition: A list of num of samples for each share.
        """
        assert n_share > 0, f"cannot split into {n_share} share"
        if self.verbose:
            log(f"  {n_sample} smp => {n_share} shards by {self.partition_mode} distr")
        if self.max_n_sample > 0:
            n_sample = min((n_sample, self.max_n_sample))
        if self.max_n_sample_per_share > 0:
            n_sample = min((n_sample, n_share * self.max_n_sample_per_share))

        if n_sample < self.min_n_sample_per_share * n_share:
            raise ValueError(f"Not enough samples. Require {self.min_n_sample_per_share} samples"
                             f" per share at least for {n_share} shares. But only {n_sample} is"
                             f" available totally.")
        n_sample -= self.min_n_sample_per_share * n_share
        if self.partition_mode == "dir":
            partition = (self.rng.dirichlet(n_share * [1]) * n_sample).astype(int)
        elif self.partition_mode == "uni":
            partition = int(n_sample // n_share) * np.ones(n_share, dtype='int')
        else:
            raise ValueError(f"Invalid partition_mode: {self.partition_mode}")

        # uniformly add residual to as many users as possible.
        for i in self.rng.choice(n_share, n_sample - np.sum(partition)):
            partition[i] += 1
            # partition[-1] += n_sample - np.sum(partition)  # add residual
        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        partition = partition + self.min_n_sample_per_share
        n_sample += self.min_n_sample_per_share * n_share
        # partition = np.minimum(partition, max_n_sample_per_share)
        partition = partition.tolist()

        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        assert len(partition) == n_share, f"{len(partition)} != {n_share}"
        return partition


class ClassWisePartitioner(Partitioner):
    def __init__(self, n_class_per_share=2, **kwargs):
        super(ClassWisePartitioner, self).__init__(**kwargs)
        self.n_class_per_share = n_class_per_share
        self._aux_partitioner = Partitioner(**kwargs)

    def __call__(self, labels, n_user, log=print):
        # reorganize labels by class
        idx_by_class = defaultdict(list)
        if len(labels) > 1e5:
            labels_iter = tqdm(labels, leave=False, desc='sort labels')
        else:
            labels_iter = labels
        for i, label in enumerate(labels_iter):
            idx_by_class[label].append(i)

        n_class = len(idx_by_class)
        assert n_user * self.n_class_per_share > n_class, f"Cannot split {n_class} classes into " \
                                                          f"{n_user} users when each user only " \
                                                          f"has {self.n_class_per_share} classes."

        # assign classes to each user.
        user_ids_by_class = defaultdict(list)
        label_sampler = shuffle_sampler(list(range(n_class)))
        for s in range(n_user):
            s_classes = [label_sampler.next() for _ in range(self.n_class_per_share)]
            for c in s_classes:
                user_ids_by_class[c].append(s)

        idx_by_user = [[] for _ in range(n_user)]
        if n_class > 100 or len(labels) > 1e5:
            idx_by_class_iter = tqdm(idx_by_class, leave=True, desc='split cls')
            log = lambda log_s: idx_by_class_iter.set_postfix_str(log_s[:10])  # tqdm.write
        else:
            idx_by_class_iter = idx_by_class
        for c in idx_by_class_iter:
            l = len(idx_by_class[c])
            log(f" class-{c} => {len(user_ids_by_class[c])} shares")
            l_by_user = self._aux_partitioner(l, len(user_ids_by_class[c]), log=log)
            base_idx = 0
            for i_user, tl in zip(user_ids_by_class[c], l_by_user):
                idx_by_user[i_user].extend(idx_by_class[c][base_idx:base_idx+tl])
                base_idx += tl
        return idx_by_user


def extract_labels(dataset: Dataset):
    if hasattr(dataset, 'targets'):
        return dataset.targets
    dl = DataLoader(dataset, batch_size=512, drop_last=False, num_workers=4, shuffle=False)
    labels = []
    dl_iter = tqdm(dl, leave=False, desc='load labels') if len(dl) > 100 else dl
    for _, targets in dl_iter:
        labels.extend(targets.cpu().numpy().tolist())
    return labels


def test_class_partitioner():
    print(f"\n==== Extract from random labels =====")
    split = ClassWisePartitioner()
    n_class = 10
    n_sample = 1000
    n_user = 100
    labels = np.random.randint(0, n_class, n_sample)
    idx_by_user = split(labels, n_user)
    _n_smp = 0
    for u in range(n_user):
        u_labels = labels[idx_by_user[u]]
        u_classes = np.unique(u_labels)
        print(f"user-{u} | {len(idx_by_user[u])} samples | {len(u_classes)} classes: {u_classes}")
        assert len(u_classes) == 2
        _n_smp += len(u_labels)
    assert _n_smp == n_sample

    print(f"\n==== Extract from dataset =====")
    from .data_loader import CifarDataset
    ds = CifarDataset('cifar10', transform=transforms.ToTensor())
    labels = extract_labels(ds)
    n_sample = len(labels)
    idx_by_user = split(labels, n_user)
    labels = np.array(labels)
    _n_smp = 0
    for u in range(n_user):
        u_labels = labels[idx_by_user[u]]
        u_classes = np.unique(u_labels)
        print(f"user-{u} | {len(idx_by_user[u])} samples | {len(u_classes)} classes: {u_classes}")
        assert len(u_classes) == 2
        _n_smp += len(u_labels)
    assert _n_smp == n_sample, f"Expected {n_sample} samples but got {_n_smp}"


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', type=str, default='none', choices=['Cifar10'],
                        help='Download datasets.')
    parser.add_argument('--test', action='store_true', help='Run test')
    args = parser.parse_args()
    if args.test:
        test_class_partitioner()
    else:
        if args.download == 'Cifar10':
            CifarDataset(download=True, train=True)
            CifarDataset(download=True, train=False)
        else:
            print(f"Nothing to download for dataset: {args.download}")
