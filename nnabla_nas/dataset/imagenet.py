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
from pathlib import Path

from nnabla import random
from nnabla_ext.cuda.experimental import dali_iterator
from sklearn.model_selection import train_test_split

from .dataloader import BaseDataLoader

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types


_pixel_mean = [255 * x for x in (0.485, 0.456, 0.406)]
_pixel_std = [255 * x for x in (0.229, 0.224, 0.225)]


@pipeline_def
def train_pipeline(image_dir, file_list, shard_id, nvjpeg_padding, num_shards=1,
                   channel_last=True, dtype="half", pad_output=False):
    jpegs, labels = fn.readers.file(file_root=image_dir, file_list=file_list,
                                    random_shuffle=True, num_shards=num_shards,
                                    shard_id=shard_id, name="Reader")
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB,
                               device_memory_padding=nvjpeg_padding,
                               host_memory_padding=nvjpeg_padding)

    images = fn.random_resized_crop(images, device="gpu", size=(224, 224))

    images = fn.crop_mirror_normalize(images, device="gpu",
                                      dtype=types.FLOAT16
                                      if dtype == "half"
                                      else types.FLOAT,
                                      output_layout=types.NHWC
                                      if channel_last else types.NCHW,
                                      crop=(224, 224),
                                      mean=_pixel_mean,
                                      std=_pixel_std,
                                      pad_output=pad_output,
                                      mirror=fn.coin_flip(probability=0.5))
    return images, labels.gpu()


@pipeline_def
def train_pipeline_color_twist(image_dir, file_list, shard_id, nvjpeg_padding, num_shards=1,
                               channel_last=True, dtype="half", pad_output=False):
    jpegs, labels = fn.readers.file(file_root=image_dir, file_list=file_list,
                                    random_shuffle=True, num_shards=num_shards,
                                    shard_id=shard_id, name="Reader")
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB,
                               device_memory_padding=nvjpeg_padding,
                               host_memory_padding=nvjpeg_padding)

    images = fn.random_resized_crop(images, device="gpu", size=(224, 224))

    images = fn.color_twist(images, device="gpu",
                            brightness=fn.random.uniform(
                                range=(1 - 32. / 255., 1 + 32. / 255)),
                            contrast=fn.random.uniform(range=(0, 1)),
                            saturation=fn.random.uniform(range=(0.5, 1 + 0.5)),
                            hue=fn.random.uniform(range=(0, 1))
                            )

    images = fn.crop_mirror_normalize(images, device="gpu",
                                      dtype=types.FLOAT16
                                      if dtype == "half"
                                      else types.FLOAT,
                                      output_layout=types.NHWC
                                      if channel_last else types.NCHW,
                                      crop=(224, 224),
                                      mean=_pixel_mean,
                                      std=_pixel_std,
                                      pad_output=pad_output,
                                      mirror=fn.random.coin_flip(probability=0.5))
    return images, labels.gpu()


@pipeline_def
def val_pipeline(image_dir, file_list, shard_id, nvjpeg_padding, num_shards=1,
                 channel_last=True, dtype="half", pad_output=False):
    jpegs, labels = fn.readers.file(file_root=image_dir, file_list=file_list,
                                    random_shuffle=True, num_shards=num_shards,
                                    shard_id=shard_id, name="Reader")
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB,
                               device_memory_padding=nvjpeg_padding,
                               host_memory_padding=nvjpeg_padding)

    images = fn.resize(images, device="gpu", resize_shorter=256)

    images = fn.crop_mirror_normalize(images, device="gpu",
                                      dtype=types.FLOAT16 if
                                      dtype == "half" else types.FLOAT,
                                      output_layout=types.NHWC
                                      if channel_last else types.NCHW,
                                      crop=(224, 224),
                                      mean=_pixel_mean,
                                      std=_pixel_std,
                                      pad_output=pad_output)
    return images, labels.gpu()


def get_data_iterators(batch_size,
                       dali_num_threads,
                       train_dir,
                       file_list,
                       dali_nvjpeg_memory_padding,
                       type_config,
                       channel_last,
                       comm,
                       training=True,
                       colortwist=False):
    r'''Creates and returns DALI data iterators

    The datasets are partitioned in distributed training
    mode according to comm rank and number of processes.
    '''
    pipe_name = train_pipeline_color_twist if (training and colortwist) else\
        train_pipeline if training else val_pipeline
    # Pipelines and Iterators for training
    pipe = pipe_name(image_dir=train_dir,
                     file_list=file_list,
                     shard_id=comm.rank,
                     nvjpeg_padding=dali_nvjpeg_memory_padding,
                     num_shards=comm.n_procs,
                     channel_last=channel_last,
                     dtype=type_config,
                     batch_size=batch_size,
                     num_threads=dali_num_threads,
                     seed=comm.rank + 1,
                     device_id=comm.rank)

    data = dali_iterator.DaliIterator(pipe)
    data.size = pipe.epoch_size("Reader") // comm.n_procs

    return data


class DataLoader(BaseDataLoader):
    def __init__(self, batch_size=1, searching=False, training=False, datapath=None,
                 train_path=None, train_file=None, valid_path=None, valid_file=None,
                 train_portion=1.0, *, augment_valid=True, colortwist=False,
                 rng=None, communicator=None, type_config=float, channel_last=False):
        r"""Dataloader for ImageNet.

        Args:
            batch_size (int, optional): The mini-batch size. Defaults to 1.
            searching (bool, optional): If `True`, the training data will be split into two parts.
                First part will be used for training the model parameters. The second part will be
                used to update the architecture parameters. Defaults to False.
            training (bool, optional): Whether training is `True`. Defaults to False.
            train_path (str, optional): Path to training images. Defaults to None.
            train_file (str, optional): A file containing training images names.
                Defaults to None.
            valid_path (str, optional): Path to validation images. Defaults to None.
            valid_file (str, optional): A file containing validation image name.
                Defaults to None.
            train_portion (float, optional): Portion of data is taken to use as training data. The rest
                will be used for validation. Defaults to 1.0. This is only considered when searching is `True`.
            augment_valid (bool, optional): Whether to use data augmentation of the validation set. Has only
                effect if searching is True. Defaults to True.
            rng (:obj:`numpy.random.RandomState`), optional): Numpy random number generator.
                Defaults to None.
            communicator (Communicator, optional): The communicator is used to support distributed learning.
                Defaults to None.
            type_config (type, optional): Configuration type. Defaults to `float`.
            channel_last(bool, optional): If True, the last dimension is
                considered as channel dimension, a.k.a NHWC order. Defaults to`False`.
        """
        self.rng = rng or random.prng  # np.random.RandomState(313) #

        train_path = datapath + train_path
        train_file = datapath + train_file
        valid_path = datapath + valid_path
        valid_file = datapath + valid_file

        if searching:
            train_file, valid_file = self._split_data(
                train_file, train_portion)
            valid_path = train_path

        self._data = get_data_iterators(
            batch_size=batch_size,
            dali_num_threads=8,
            train_dir=train_path if training else valid_path,
            file_list=train_file if training else valid_file,
            dali_nvjpeg_memory_padding=64 * (1 << 20),
            type_config=type_config,
            channel_last=channel_last,
            comm=communicator,
            training=training or (searching and augment_valid),
            colortwist=colortwist
        )

    def _split_data(self, file, portion):
        # split training data into train and valid
        list_files, labels = [], []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line != '':
                    list_files.append(line)
                    labels.append(line.split(' ')[1])

        train_size = int(len(labels) * portion)
        train, valid = train_test_split(list_files, stratify=labels,
                                        train_size=train_size,
                                        random_state=self.rng)
        path = os.path.join('__nnabla_nas__', 'imagenet_{}'.format(portion))

        Path(path).mkdir(parents=True, exist_ok=True)
        train_file = os.path.join(path, 'train.txt')
        valid_file = os.path.join(path, 'val.txt')

        if not os.path.exists(train_file):
            with open(train_file, 'w') as f:
                for item in train:
                    f.write("%s\n" % item)

        if not os.path.exists(valid_file):
            with open(valid_file, 'w') as f:
                for item in valid:
                    f.write("%s\n" % item)

        return train_file, valid_file

    def __len__(self):
        return self._data.size

    def next(self):
        x, y = self._data.next()
        return {"inputs": [x], "targets": [y]}
