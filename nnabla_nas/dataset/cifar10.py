# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tarfile

from nnabla import random
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download
import numpy as np
from sklearn.model_selection import train_test_split

from .dataloader import BaseDataLoader
from ..utils.data import transforms


def download_data(train=True):
    data_uri = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    logger.info('Getting labeled data from {}.'.format(data_uri))

    r = download(data_uri)  # file object returned
    with tarfile.open(fileobj=r, mode="r:gz") as fpin:
        if train:
            images = []
            labels = []
            for member in fpin.getmembers():
                if "data_batch" not in member.name:
                    continue
                fp = fpin.extractfile(member)
                data = np.load(fp, encoding="bytes", allow_pickle=True)
                images.append(data[b"data"])
                labels.append(data[b"labels"])
            size = 50000
            images = np.concatenate(images).reshape(size, 3, 32, 32)
            labels = np.concatenate(labels).reshape(-1, 1)
        else:
            for member in fpin.getmembers():
                if "test_batch" not in member.name:
                    continue
                fp = fpin.extractfile(member)
                data = np.load(fp, encoding="bytes", allow_pickle=True)
                images = data[b"data"].reshape(10000, 3, 32, 32)
                labels = np.array(data[b"labels"]).reshape(-1, 1)
    return (images, labels)


class CifarDataSource(DataSource):

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, images, labels, shuffle=False, rng=None):
        super(CifarDataSource, self).__init__(shuffle=shuffle, rng=rng)
        self._train = True
        self._images = images
        self._labels = labels
        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(CifarDataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()


def get_data(train, comm, rng):
    # download the data
    images, labels = download_data(train)

    n = len(labels)
    if rng is None:
        rng = random.prng

    if train:
        index = rng.randint(0, n, size=n)
    else:
        index = np.arange(n)

    num = n // comm.n_procs

    selected_idx = index[num*comm.rank:num*(comm.rank + 1)]

    return images[selected_idx], labels[selected_idx]


class DataLoader(BaseDataLoader):
    r"""DataLoader for cifar10.

    Args:
        batch_size (int, optional): The mini-batch size. Defaults to 1.
        searching (bool, optional): If `True`, the training data will be split into two parts.
            First part will be used for training the model parameters. The second part will be
            used to update the architecture parameters. Defaults to False.
        training (bool, optional): Whether training is `True`. Defaults to False.
        train_portion (float, optional): Portion of data is taken to use as training data. The rest
            will be used for validation. Defaults to 1.0. This is only considered when searching is `True`.
        rng (:obj:`numpy.random.RandomState`), optional): Numpy random number generator.
            Defaults to None.
        communicator (Communicator, optional): The communicator is used to support distributed
            learning. Defaults to None.
    """

    def __init__(self, batch_size=1, searching=False, training=False,
                 train_portion=1.0, rng=None, communicator=None, *args):
        rng = rng or random.prng

        if searching:
            images, labels = get_data(True, communicator, rng)
            train_size = int(len(labels) * train_portion)
            data = train_test_split(images, labels, stratify=labels,
                                    train_size=train_size, random_state=rng)
            idx = 0 if training else 1
            X, y = data[idx], data[idx + 2]
        else:
            X, y = get_data(training, communicator, rng)

        self._data = data_iterator(
            CifarDataSource(X, y, shuffle=searching or training, rng=rng),
            batch_size=batch_size,
            rng=rng,
            with_memory_cache=False,
            with_file_cache=False
        )

    def __len__(self):
        return self._data.size

    def next(self):
        x, y = self._data.next()
        return {"inputs": [x], "targets": [y]}

    def transform(self, key='train'):
        r"""Return a transform applied to data augmentation."""
        assert key in ('train', 'valid')

        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)
        scale = 1./255.0
        pad_width = (4, 4, 4, 4)

        if key == 'train':
            return transforms.Compose([
                transforms.Cutout(8, prob=1, seed=123),
                transforms.Normalize(mean=mean, std=std, scale=scale),
                transforms.RandomCrop((3, 32, 32), pad_width=pad_width),
                transforms.RandomHorizontalFlip()
            ])

        return transforms.Compose([
            transforms.Normalize(mean=mean, std=std, scale=scale)
        ])
