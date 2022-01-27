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

import numpy as np

from nnabla import random
from nnabla_ext.cuda.experimental import dali_iterator
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
from sklearn.model_selection import train_test_split
import sys
import traceback

import math
from .dataloader import BaseDataLoader
from ..contrib.classification.ofa.ofa_modules.my_random_resize_crop import MyResize
from ..utils.data.transforms import Compose

_pixel_mean = [255 * x for x in (0.485, 0.456, 0.406)]
_pixel_std = [255 * x for x in (0.229, 0.224, 0.225)]

def int_div_ceil(a, b):
    '''
    returns int(ceil(a / b))
    '''
    return (a + b - 1) // b

class TrainPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, shard_id, image_dir, file_list,
                 nvjpeg_padding, prefetch_queue=3, seed=1, num_shards=1,
                 channel_last=True, dtype="half", pad_output=False):
        super(TrainPipeline, self).__init__(
            batch_size, num_threads, shard_id, seed=seed,
            prefetch_queue_depth=prefetch_queue)
        self.input = ops.FileReader(file_root=image_dir, file_list=file_list,
                                    random_shuffle=True, num_shards=num_shards,
                                    shard_id=shard_id)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB,
                                       device_memory_padding=nvjpeg_padding,
                                       host_memory_padding=nvjpeg_padding)
        self.rrc = ops.RandomResizedCrop(device="gpu", size=(224, 224))
        self.colortwist = ops.ColorTwist(device="gpu")
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16
                                            if dtype == "half"
                                            else types.FLOAT,
                                            output_layout=types.NHWC
                                            if channel_last else types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=_pixel_mean,
                                            std=_pixel_std,
                                            pad_output=pad_output)
        self.coin = ops.CoinFlip(probability=0.5)
        self.brightness = ops.Uniform(range=(1 - 32./255., 1 + 32./255))
        self.contrast = ops.Uniform(range=(0, 1))
        self.saturation = ops.Uniform(range=(0.5, 1+0.5))
        self.hue = ops.Uniform(range=(0, 1))

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.rrc(images)
        images = self.colortwist(
            images, 
            brightness=self.brightness(), 
            contrast=self.contrast(), 
            saturation=self.saturation(),
            hue=self.hue())
        images = self.cmnp(images, mirror=self.coin())
        return images, labels.gpu()

class ValPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, shard_id, image_dir, file_list,
                 nvjpeg_padding, seed=1, num_shards=1, channel_last=True,
                 dtype='half', pad_output=False):
        super(ValPipeline, self).__init__(
            batch_size, num_threads, shard_id, seed=seed)
        self.input = ops.FileReader(file_root=image_dir, file_list=file_list,
                                    random_shuffle=False,
                                    num_shards=num_shards, shard_id=shard_id)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB,
                                       device_memory_padding=nvjpeg_padding,
                                       host_memory_padding=nvjpeg_padding)
        self.res = ops.Resize(device="gpu", resize_shorter=256)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if
                                            dtype == "half" else types.FLOAT,
                                            output_layout=types.NHWC
                                            if channel_last else types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=_pixel_mean,
                                            std=_pixel_std,
                                            pad_output=pad_output)

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        images = self.cmnp(images)
        return images, labels.gpu()


def get_data_iterators(batch_size,
                       dali_num_threads,
                       train_dir,
                       file_list,
                       dali_nvjpeg_memory_padding,
                       type_config,
                       channel_last,
                       comm,
                       training=True):
    r'''Creates and returns DALI data iterators

    The datasets are partitioned in distributed training
    mode according to comm rank and number of processes.
    '''
    cls_name = TrainPipeline if training else ValPipeline
    # Pipelines and Iterators for training
    train_pipe = cls_name(batch_size, dali_num_threads, comm.rank,
                          train_dir,
                          file_list,
                          dali_nvjpeg_memory_padding,
                          seed=comm.rank + 1,
                          num_shards=comm.n_procs,
                          channel_last=channel_last,
                          dtype=type_config)

    try:
        data = dali_iterator.DaliIterator(train_pipe)
        #data.size = train_pipe.epoch_size("Reader") // comm.n_procs
        data.size = int_div_ceil(train_pipe.epoch_size("Reader"), comm.n_procs)
    except Exception as e:
        print(f"Exception: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

    return data

class DataLoader(BaseDataLoader):
    def __init__(self, batch_size=1, searching=False, valid_size=10000, training=False, 
                 train_path=None, train_file=None, valid_path=None, valid_file=None,
                 train_portion=1.0, rng=None, communicator=None, type_config=float, *args):
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
            rng (:obj:`numpy.random.RandomState`), optional): Numpy random number generator.
                Defaults to None.
            communicator (Communicator, optional): The communicator is used to support distributed learning.
                Defaults to None.
            type_config (type, optional): Configuration type. Defaults to `float`.
        """

        self.rng = rng or random.prng

        train_path = os.environ[train_path]
        valid_path = os.environ[valid_path]
        train_file = os.environ[train_file]
        valid_file = os.environ[valid_file]

        if searching:
            #train_file, valid_file = self._split_data(train_file, train_portion)
            train_file, valid_file = self._random_sample_valid_set(train_file, valid_size)
            valid_path = train_path
        
        self._data = get_data_iterators(
            batch_size=batch_size,
            dali_num_threads=8,
            train_dir=train_path if training else valid_path,
            file_list=train_file if training else valid_file,
            dali_nvjpeg_memory_padding=64*(1 << 20),
            type_config=type_config,
            channel_last=False,
            comm=communicator,
            training=training
            #training=searching or training
        )
        # for subset sampling
        self.type_config = type_config
        self.communicator = communicator
        self.train_path = train_path
        self.train_file = train_file

    def _random_sample_valid_set(self, file, valid_size):
        # split training data into train and valid
        list_files, labels = [], []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line != '':
                    list_files.append(line)
                    labels.append(line.split(' ')[1])

        rs = np.random.RandomState(2147483647)
        rand_indexes = list(rs.permutation(len(labels)))

        valid_indexes, train_indexes = rand_indexes[:valid_size], rand_indexes[valid_size:]
        train, valid = [list_files[idx] for idx in train_indexes], [list_files[idx] for idx in valid_indexes]

        path = os.path.join('__nnabla_nas__', 'imagenet')

        Path(path).mkdir(parents=True, exist_ok=True)
        train_file = os.path.join(path, 'train.txt')
        valid_file = os.path.join(path, 'val.txt')
        
        with open(train_file, 'w') as f:
            for i, item in enumerate(train):
                if i == len(train) - 1:
                    f.write(f"{item}")
                else:
                    f.write(f"{item}\n")

        with open(valid_file, 'w') as f:
            for i, item in enumerate(valid):
                if i == len(valid) - 1:
                    f.write(f"{item}")
                else:
                    f.write(f"{item}\n")

        return train_file, valid_file

    def _get_subset_file_list(self, file, n_images):
        # split training data into train and valid
        list_files, labels = [], []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line != '':
                    list_files.append(line)
                    labels.append(line.split(' ')[1])

        rs = np.random.RandomState(937162211)
        rand_indexes = list(rs.permutation(len(labels)))
        chosen_indexes = rand_indexes[:n_images]
        subset_list_files = [list_files[idx] for idx in chosen_indexes]

        path = os.path.join('__nnabla_nas__', 'imagenet')

        Path(path).mkdir(parents=True, exist_ok=True)
        subset_file = os.path.join(path, 'subset.txt')
        
        with open(subset_file, 'w') as f:
            for i, item in enumerate(subset_list_files):
                if i == len(subset_list_files) - 1:
                    f.write(f"{item}")
                else:
                    f.write(f"{item}\n")

        return subset_file

    def __len__(self):
        return self._data.size

    def transform(self):
        r"""Return a transform.

        Args:
            key (str, optional): Type of transform. Defaults to 'train'.
        """
        #assert key in ('train', 'valid')
        """if key == 'train':
            transforms = [
                MyRandomResizedCrop(),
                RandomHorizontalFlip(),
                Normalize(_pixel_mean, _pixel_std, 1),
            ]
        else:
            transforms = [
                Resize((int(math.ceil(image_size / 0.875)), int(math.ceil(image_size / 0.875)))),
                CenterCrop(image_size),
                Normalize(_pixel_mean, _pixel_std, 1),
            ]"""
        transforms = [MyResize()]
        
        return Compose(transforms)

    def next(self):
        x, y = self._data.next()
        return {"inputs": [x], "targets": [y]}
    
    def build_sub_train_loader(self, n_images, batch_size):
        # used for resetting BN running statistics
        path = os.path.join('__nnabla_nas__', 'imagenet')
        train_path = os.path.join(path, 'train.txt')
        sub_train_path = train_path if os.path.exists(train_path) else self.train_path

        train_subset_file = self._get_subset_file_list(sub_train_path, n_images)
        #train_subset_file = self._get_subset_file_list(self.train_file, n_images)
        dataloader = get_data_iterators(
            batch_size=batch_size,
            dali_num_threads=8,
            train_dir=self.train_path,
            file_list=train_subset_file,
            dali_nvjpeg_memory_padding=64*(1 << 20),
            type_config=self.type_config,
            channel_last=False,
            comm=self.communicator,
            training=True
        )
        return dataloader

