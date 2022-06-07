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

import os
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


class CityscapesSegmentation:
    NUM_CLASSES = 19

    def __init__(self, root, split="train"):

        self.root = root
        self.split = split
        # self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def read_data(self):

        # img_path = self.files[self.split][index].rstrip()
        # lbl_path = os.path.join(self.annotations_base,
        #                         img_path.split(os.sep)[-2],
        #                         os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        images = []
        labels = []
        for img_root_path in self.files[self.split]:
            img_path = img_root_path.rstrip()
            lbl_path = os.path.join(self.annotations_base,
                                    img_path.split(os.sep)[-2],
                                    os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
            _img = Image.open(img_path).convert('RGB')
            _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
            _tmp = self.encode_segmap(_tmp)
            _target = Image.fromarray(_tmp)

            images.append(np.array(_img))
            labels.append(np.array(_target))

        img_size = _img.shape[1]
        size = len(self.files[self.split])
        images = np.concatenate(images).reshape(size, 3, img_size, img_size)
        labels = np.concatenate(labels).reshape(size, 1, img_size, img_size)

        return (images, labels)
        # sample = {'image': _img, 'label': _target}

        # if self.split == 'train':
        #     return self.transform_tr(sample)
        # elif self.split == 'val':
        #     return self.transform_val(sample)
        # elif self.split == 'test':
        #     return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    # def transform_tr(self, sample):
    #     composed_transforms = transforms.Compose([
    #         tr.RandomHorizontalFlip(),
    #         tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
    #         tr.RandomGaussianBlur(),
    #         tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])

    #     return composed_transforms(sample)

    # def transform_val(self, sample):

    #     composed_transforms = transforms.Compose([
    #         tr.FixScaleCrop(crop_size=self.args.crop_size),
    #         tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])

    #     return composed_transforms(sample)

    # def transform_ts(self, sample):

    #     composed_transforms = transforms.Compose([
    #         tr.FixedResize(size=self.args.crop_size),
    #         tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])

    #     return composed_transforms(sample)

# def download_data(train=True):
#     data_uri = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
#     logger.info('Getting labeled data from {}.'.format(data_uri))

#     r = download(data_uri)  # file object returned
#     with tarfile.open(fileobj=r, mode="r:gz") as fpin:
#         if train:
#             images = []
#             labels = []
#             for member in fpin.getmembers():
#                 if "data_batch" not in member.name:
#                     continue
#                 fp = fpin.extractfile(member)
#                 data = np.load(fp, encoding="bytes", allow_pickle=True)
#                 images.append(data[b"data"])
#                 labels.append(data[b"labels"])
#             size = 50000
#             images = np.concatenate(images).reshape(size, 3, 32, 32)
#             labels = np.concatenate(labels).reshape(-1, 1)
#         else:
#             for member in fpin.getmembers():
#                 if "test_batch" not in member.name:
#                     continue
#                 fp = fpin.extractfile(member)
#                 data = np.load(fp, encoding="bytes", allow_pickle=True)
#                 images = data[b"data"].reshape(10000, 3, 32, 32)
#                 labels = np.array(data[b"labels"]).reshape(-1, 1)
#     return (images, labels)


class CityscapesDataSource(DataSource):

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, images, labels, shuffle=False, rng=None):
        super(CityscapesDataSource, self).__init__(shuffle=shuffle, rng=rng)
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
        super(CityscapesDataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1, H, W)."""
        return self._labels.copy()


def get_data(split="train", comm, rng):
    # read the data
    cityscapes = CityscapesSegmentation(root, split)
    images, labels = cityscapes.read_data()

    n = len(labels)
    if rng is None:
        rng = random.prng

    if train:
        index = rng.permutation(n)
    else:
        index = np.arange(n)

    num = n // comm.n_procs

    selected_idx = index[num * comm.rank:num * (comm.rank + 1)]

    return images[selected_idx], labels[selected_idx]


class DataLoader(BaseDataLoader):
    r"""DataLoader for cityscapes.

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
                 train_portion=1.0, rng=None, communicator=None):
        rng = rng or random.prng
        if searching:
            images, labels = get_data("train", communicator, rng)
            train_size = int(len(labels) * train_portion)
            data = train_test_split(images, labels, stratify=labels,
                                    train_size=train_size, random_state=rng)
            idx = 0 if training else 1
            X, y = data[idx], data[idx + 2]
        else:
            if training:
                split = "train"
            else:
                split = "val"
            X, y = get_data(split, communicator, rng)

        self._data = data_iterator(
            CityscapesDataSource(X, y, shuffle=searching or training, rng=rng),
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

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        scale = 1. / 255.0
        pad_width = (4, 4, 4, 4)

        if key == 'train':
            return transforms.Compose([
                transforms.Cutout(8, prob=1, seed=123),
                transforms.Normalize(mean=mean, std=std, scale=scale),
                transforms.RandomCrop((3, 513, 513), pad_width=pad_width),
                transforms.RandomHorizontalFlip()
            ])

        return transforms.Compose([
            transforms.Normalize(mean=mean, std=std, scale=scale)
        ])
