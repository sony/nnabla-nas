# Copyright (c) 2021 Sony Group Corporation. All Rights Reserved.
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

from nnabla import random
import nnabla.utils.communicator_util as commutil
from nnabla.utils.load import _create_dataset

from .dataloader import BaseDataLoader
from ..utils.data import transforms


def get_sliced_data_iterator(dataset, comm, training, portion):
    it = dataset.data_iterator()

    size = it._data_source.size // comm.n_procs
    start = size * comm.rank
    end = size * (comm.rank + 1)
    if portion:
        train_size = int(size * portion)
        if training:
            end = start + train_size
        else:
            start += train_size

    return it.slice(rng=it._rng, slice_start=start, slice_end=end)


class DataLoader(BaseDataLoader):
    r"""DataLoader for csv.

    Args:
        batch_size (int, optional): The mini-batch size. Defaults to 1.
        searching (bool, optional): If `True`, the training data will be split into two parts.
            First part will be used for training the model parameters. The second part will be
            used to update the architecture parameters. Defaults to False.
        training (bool, optional): Whether training is `True`. Defaults to False.
        training_file (str, optional): Path to training csv file. Defaults to None.
        valid_file (str, optional): Path to validation csv file. Defaults to None.
        training_cache_dir (str, optional): Path to training cache directory. Defaults to None.
        valid_cache_dir (str, optional): Path to validation cache directory. Defaults to None.
        train_portion (float, optional): Portion of data is taken to use as training data. The rest
            will be used for validation. Defaults to 1.0. This is only considered when searching is `True`.
        rng (:obj:`numpy.random.RandomState`), optional): Numpy random number generator.
            Defaults to None.
        augmentation (dict, optional): Information on how to augment. Defaults to None.
        communicator (Communicator, optional): The communicator is used to support distributed
            learning. Defaults to None.
    """

    def __init__(self, batch_size=1, searching=False, training=False,
                 train_file=None, valid_file=None,
                 train_cache_dir=None, valid_cache_dir=None,
                 train_portion=1.0, rng=None, augmentation=None,
                 communicator=None):
        self.rng = rng or random.prng
        self.augmentation = augmentation

        if searching:
            file = train_file
            cache_dir = train_cache_dir
            shuffle = True
            portion = train_portion
        else:
            file = train_file if training else valid_file
            cache_dir = train_cache_dir if training else valid_cache_dir
            shuffle = training
            portion = None

        commutil._current_communicator = communicator.comm
        dataset = _create_dataset(
            file,
            batch_size,
            shuffle=shuffle,
            no_image_normalization=True,
            cache_dir=cache_dir,
            overwrite_cache=False,
            create_cache_explicitly=False,
            prepare_data_iterator=True,
            dataset_index=313)
        self._data = get_sliced_data_iterator(
            dataset, communicator, training, portion)

    def __len__(self):
        return self._data.size

    def next(self):
        x, y = self._data.next()
        return {"inputs": [x], "targets": [y]}

    def transform(self, key='train'):
        r"""Return a transform applied to data augmentation."""
        assert key in ('train', 'valid')

        if self.augmentation:
            type = self.augmentation.get('type')
            norm = self.augmentation.get('normalize')
        else:
            type = None
            norm = None

        if type == 'cifar10':
            mean = (0.49139968, 0.48215827, 0.44653124)
            std = (0.24703233, 0.24348505, 0.26158768)
            scale = 1./255.0
        elif type == 'imagenet':
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            scale = 1./255.0
        else:
            mean = (0.0, 0.0, 0.0)
            std = (1.0, 1.0, 1.0)
            scale = 1./255.0

        if key == 'train':
            if type == 'cifar10':
                pad_width = (4, 4, 4, 4)
                return transforms.Compose([
                    transforms.Cutout(8, prob=1, seed=123),
                    transforms.Normalize(mean=mean, std=std, scale=scale),
                    transforms.RandomCrop((3, 32, 32), pad_width=pad_width),
                    transforms.RandomHorizontalFlip()
                ])
            elif type == 'imagenet':
                return transforms.Compose([
                    transforms.Normalize(mean=mean, std=std, scale=scale),
                    transforms.RandomResizedCrop((3, 224, 224),
                                                 scale=(1.0, 2.3), ratio=1.33),
                    transforms.RandomHorizontalFlip()
                ])
            else:
                pass  # same as valid

        if type == 'cifar10' or norm:
            return transforms.Compose([
                transforms.Normalize(mean=mean, std=std, scale=scale)
            ])
        elif type == 'imagenet':
            return transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.Normalize(mean=mean, std=std, scale=scale)
            ])
        else:
            return transforms.Compose([])
