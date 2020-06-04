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
from collections import OrderedDict
from pathlib import Path

import nnabla as nn
import nnabla.utils.learning_rate_scheduler as LRS
from nnabla import random
from nnabla.logger import logger
from sklearn.model_selection import train_test_split

from nnabla_nas.utils.data.dataloader import BaseDataLoader
from nnabla_nas.utils.data import transforms
from nnabla_nas.optimizer import Optimizer
from nnabla_nas.utils import estimator as EST
from nnabla_nas.utils import helper


class Configuration(object):

    def __init__(self, conf=None):
        conf = conf or dict()

        # data set
        conf['dataset'] = conf.get('dataset', 'cifar10')
        self.dataset = conf['dataset']

        # training epochs
        conf['epoch'] = conf.get('epoch', 150)
        self.epoch = conf['epoch']

        # batchsize training
        conf['batch_size_train'] = conf.get('batch_size_train', 256)
        self.bs_train = conf['batch_size_train']

        # batchsize valid
        conf['batch_size_valid'] = conf.get('batch_size_valid', 256)
        self.bs_valid = conf['batch_size_valid']

        # mini batchsize training
        conf['mini_batch_train'] = conf.get('mini_batch_train', 64)
        self.mbs_train = conf['mini_batch_train']

        # mini batchsize valid
        conf['mini_batch_valid'] = conf.get('mini_batch_valid', 64)
        self.mbs_valid = conf['mini_batch_valid']

        # input shape
        conf['input_shape'] = conf.get('input_shape', (3, 32, 32))
        self.input_shape = conf['input_shape']

        assert self.bs_train % self.mbs_train == 0
        assert self.bs_valid % self.mbs_valid == 0

        # number of epochs for warming up
        conf['warmup'] = conf.get('warmup', 0)
        self.warmup = conf['warmup']

        # frequency of messages
        conf['print_frequency'] = conf.get('print_frequency', 20)
        self.print_frequency = conf['print_frequency']

        # training portion
        conf['train_portion'] = conf.get('train_portion', 1)
        self.train_portion = conf['train_portion']

        # output path
        conf['output_path'] = conf.get('output_path', 'log')
        self.output_path = conf['output_path']

        # cutout
        conf['cutout'] = conf.get('cutout', 0)
        self.cutout = conf['cutout']

        # auxiliar weights
        conf['aux_weight'] = conf.get('aux_weight', 0)
        self.aux_weight = conf['aux_weight']

        self.conf = conf

    def parse(self):
        r"""Returns a dict containing options for a Runner."""
        conf = self.conf.copy()
        options = dict()

        # define constraints
        parser = RegularizerParser(self)
        options['regularizer'] = parser.parse(conf.get('regularizer', dict()))
        conf['regularizer'] = parser.summary().copy()

        # define dataloader for training and validating
        parser = DataloaderParser(self)
        options['dataloader'] = parser.parse(conf)

        # define data augmentation for training and validating
        parser = TransformParser(self)
        options['transform'] = parser.parse(conf)

        # define optimizer
        parser = OptimizerParser(self, len(options['dataloader']['train']))
        options['optimizer'] = parser.parse(conf.get('optimizer', dict()))
        conf['optimizer'] = parser.summary().copy()

        # a placeholder to store input and output variables
        parser = PlaceholderParser(self)
        options['placeholder'] = parser.parse(conf.get('placeholder', dict()))

        Path(conf['output_path']).mkdir(parents=True, exist_ok=True)
        file = os.path.join(conf['output_path'], 'config.json')
        if self.conf['comm'].rank == 0:
            logger.info(f'Saving the configurations to {file}')
            helper.write_to_json_file(conf, file)

        return options


class OptionParser(object):
    def __init__(self, options):
        self.options = options
        self.conf = dict()

    def parse(self, args: dict):
        raise NotImplementedError

    def summary(self):
        return self.conf


class OptimizerParser(OptionParser):
    def __init__(self, configuration, n):
        self.conf = configuration
        self.iter_train = n * self.conf.epoch // self.conf.bs_train
        self.iter_warm = n * self.conf.warmup // self.conf.bs_train

    def parse(self, args):
        conf = args.copy()

        optimizer = dict()
        for key in ['train', 'valid', 'warmup']:
            conf[key] = conf.get(key, dict())
            optim = conf[key]
            optim['solver'] = optim.get('solver', dict())
            optim['lr_scheduler'] = optim.get('lr_scheduler', None)

            lr_scheduler = None

            if optim['lr_scheduler'] is not None:
                scheduler_params = optim.pop('lr_scheduler').copy()
                class_name = scheduler_params.pop('name')
                lr_scheduler = LRS.__dict__[class_name](**scheduler_params)
            elif key != 'valid':
                optim['lr_scheduler'] = dict()
                optim['lr_scheduler']['name'] = 'CosineScheduler'
                optim['lr_scheduler']['lr'] = optim['solver'].get('lr', 0.01)
                optim['lr_scheduler']['max_iter'] = (
                    self.iter_train if key == 'train' else self.iter_warm
                )
                lr_scheduler = LRS.__dict__['CosineScheduler'](
                    init_lr=optim['lr_scheduler']['lr'],
                    max_iter=optim['lr_scheduler']['max_iter']
                )
            args = optim.copy()
            optim['solver']['name'] = optim['solver'].get('name', 'Sgd')
            sol = optim['solver'].copy()

            optimizer[key] = Optimizer(
                retain_state=True,
                weight_decay=args.pop('weight_decay', None),
                grad_clip=args.pop('grad_clip', None),
                lr_scheduler=lr_scheduler,
                name=sol.pop('name', 'Sgd'), **sol
            )
        self.conf = conf
        return optimizer


class RegularizerParser(OptionParser):

    def parse(self, args):
        conf = args.copy()
        regularizer = dict()
        for k, v in args.items():
            opt = v.copy()
            regularizer[k] = dict()
            regularizer[k]['bound'] = opt.pop('bound', 0)
            regularizer[k]['weight'] = opt.pop('weight', 0)
            v['bound'] = regularizer[k]['bound']
            v['weight'] = regularizer[k]['weight']
            v['name'] = opt.pop('name', 'LatencyEstimator')
            regularizer[k]['reg'] = EST.__dict__[v['name']](**opt)
        self.conf = conf
        return regularizer


class PlaceholderParser(OptionParser):

    def parse(self, conf):
        opts = self.options
        placeholder = OrderedDict({
            'train': {
                'input': nn.Variable((opts.mbs_train, *opts.input_shape)),
                'target': nn.Variable((opts.mbs_train, 1))
            },
            'valid': {
                'input': nn.Variable((opts.mbs_valid, *opts.input_shape)),
                'target': nn.Variable((opts.mbs_valid, 1))
            }
        })
        return placeholder


class DataloaderParser(OptionParser):

    def parse(self, conf):
        opts = self.options
        if opts.dataset == 'imagenet':
            from nnabla_nas.dataset.imagenet.imagenet import get_data_iterators

            train_path = '/speech/db/Images/ILSVRC-2012/img_train/'
            valid_path = '/speech/db/Images/ILSVRC-2012/img_val'

            train_file = '/speech/db/Images/ILSVRC-2012/train.txt'
            valid_file = '/speech/db/Images/ILSVRC-2012/val.txt'

            if conf['search']:
                # split training data into train and valid
                list_files, labels = [], []
                with open(train_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line != '':
                            list_files.append(line)
                            labels.append(line.split(' ')[1])

                train_size = int(len(labels) * opts.train_portion)
                train, valid = train_test_split(list_files, stratify=labels,
                                                train_size=train_size,
                                                random_state=random.prng)

                Path(opts.output_path).mkdir(parents=True, exist_ok=True)
                train_file = os.path.join(opts.output_path, 'train.txt')
                valid_file = os.path.join(opts.output_path, 'val.txt')
                valid_path = train_path

                with open(train_file, 'w') as f:
                    for item in train:
                        f.write("%s\n" % item)

                with open(valid_file, 'w') as f:
                    for item in valid:
                        f.write("%s\n" % item)

            tdata = get_data_iterators(
                batch_size=opts.mbs_train,
                dali_num_threads=8,
                train_dir=train_path,
                file_list=train_file,
                dali_nvjpeg_memory_padding=64*(1 << 20),
                type_config=float,
                channel_last=False,
                comm=conf['comm'],
                stream_event_handler=conf['event'],
                training=True
            )
            vdata = get_data_iterators(
                batch_size=opts.mbs_valid,
                dali_num_threads=8,
                train_dir=valid_path,
                file_list=valid_file,
                dali_nvjpeg_memory_padding=64*(1 << 20),
                type_config=float,
                channel_last=False,
                comm=conf['comm'],
                stream_event_handler=conf['event'],
                training=False
            )

            return {
                'train': DataLoader(tdata),
                'valid': DataLoader(vdata)
            }
        elif opts.dataset == 'cifar10':
            from nnabla_nas.dataset.cifar10.cifar10 import get_data_iterators
            # dataset configuration
            data_train, data_valid = get_data_iterators(
                batch_size=(opts.mbs_train, opts.mbs_valid),
                train=conf['search'],
                comm=conf['comm'],
                portion=opts.train_portion
            )
            train_transform, valid_transform = helper.dataset_transformer(conf)
            return {
                'train': DataLoader(data_train, train_transform),
                'valid': DataLoader(data_valid, valid_transform)
            }

        raise f'{opts.dataset} is not supported'


class TransformParser(OptionParser):

    def parse(self, conf):
        # TODO: setup data augmentation from config files
        opts = self.options
        if opts.dataset == 'imagenet':
            return {
                'train': transforms.Compose([]),
                'valid': transforms.Compose([])
            }

        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)
        scale = 1./255.0
        pad_width = (4, 4, 4, 4)
        return {
            'train': transforms.Compose([
                transforms.Normalize(mean=mean, std=std, scale=scale),
                transforms.RandomCrop(opts.input_shape, pad_width=pad_width),
                transforms.RandomHorizontalFlip()
            ]),
            'valid': transforms.Compose([
                transforms.Normalize(mean=mean, std=std, scale=scale)
            ])
        }
