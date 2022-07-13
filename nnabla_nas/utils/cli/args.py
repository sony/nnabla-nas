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

import sys

import nnabla.utils.learning_rate_scheduler as LRS

from nnabla_nas import dataset
from nnabla_nas.optimizer import Optimizer
from nnabla_nas.utils import estimator as EST
from nnabla_nas.utils.learning_rate_scheduler import CosineSchedulerWarmup


class Configuration(object):
    r"""A Configuration class base.

    Args:
        conf (dict): A dictionary containing configuration.
    """

    def __init__(self, conf):
        self.hparams = self.get_hyperparameters(conf['hparams'])
        self.dataloader = self.get_dataloader(conf['dataloader'])
        self.optimizer = self.get_optimizer(conf['optimizer'])
        # optional configuration
        self.regularizer = self.get_regularizer(conf.get('regularizer', dict()))

    def get_hyperparameters(self, conf):
        r"""Setup hyperparameters."""
        hparams = {  # defaults hyper-parameters
            "batch_size_train": 64,
            "batch_size_valid": 64,
            "mini_batch_train": 16,
            "mini_batch_valid": 16,
            "print_frequency": 20,
            "warmup": 0,
            "epoch": 50,
            "loss_weights": None,
            "input_shapes": [
                [3, 32, 32]
            ]
        }
        hparams.update(conf)

        # check validity of batch sizes
        assert hparams['batch_size_train'] % hparams['mini_batch_train'] == 0
        assert hparams['batch_size_valid'] % hparams['mini_batch_valid'] == 0

        return hparams

    def get_dataloader(self, conf):
        r"""Setup dataloader."""
        assert len(conf) == 1

        name, args = list(conf.items())[0]
        try:
            loader_cls = dataset.__dict__[name].DataLoader
        except ModuleNotFoundError:
            print(f"dataset `{name}` is not supported.")
            sys.exit(-1)
        args.update({
            'communicator': self.hparams['comm']
        })
        return {
            'train': loader_cls(
                searching=self.hparams['search'],
                training=True,
                batch_size=self.hparams['mini_batch_train'],
                ** args),
            'valid': loader_cls(
                searching=self.hparams['search'],
                training=False,
                batch_size=self.hparams['mini_batch_valid'],
                ** args),
            'test': loader_cls(
                searching=False,
                training=False,
                batch_size=self.hparams['mini_batch_valid'],
                ** args)
        }

    def get_optimizer(self, conf):
        r"""Setup optimizer."""
        optimizer = dict()
        for name, args in conf.items():
            try:
                lr_scheduler = None
                if 'lr_scheduler' in args:
                    class_name = args['lr_scheduler']
                    try:
                        lr = args['lr']
                    except KeyError:
                        lr = args['alpha']  # for adam
                    bz = self.hparams['batch_size_train'if name != 'valid' else 'batch_size_valid']
                    # epoch = self.hparams['epoch'] if name == 'train' else self.hparams['warmup']
                    epoch = self.hparams['epoch'] if 'train' in name else self.hparams['warmup']
                    max_iter = epoch * \
                        len(self.dataloader['valid' if name == 'valid' else 'train']) // bz
                    if class_name == "CosineSchedulerWarmup":
                        batch_iters = len(self.dataloader['valid' if name == 'valid' else 'train']) // bz
                        warmup_iter = self.hparams['cosine_warmup_epoch'] * batch_iters
                        if self.hparams['warmup_lr'] < 0:
                            warmup_lr = args['lr']
                        else:
                            warmup_lr = self.hparams['warmup_lr']
                        lr_scheduler = CosineSchedulerWarmup(
                            base_lr=lr, max_iter=max_iter, warmup_iter=warmup_iter, warmup_lr=warmup_lr)
                    elif class_name == "StepScheduler":
                        decay_rate = self.hparams["step_decay_rate"]
                        batch_iters = len(self.dataloader['valid' if name == 'valid' else 'train']) // bz
                        epoch_steps = self.hparams["epoch_steps"]  # number of epochs before each decay in lr
                        iter_steps = [ep * batch_iters for ep in range(epoch_steps, epoch+1, epoch_steps)]
                        lr_scheduler = LRS.StepScheduler(init_lr=lr, gamma=decay_rate, iter_steps=iter_steps)
                    else:
                        lr_scheduler = LRS.__dict__[class_name](init_lr=lr, max_iter=max_iter)
                args['lr_scheduler'] = lr_scheduler
                optimizer[name] = Optimizer(**args)
            except ModuleNotFoundError:
                print(f"optimizer `{name}` is not supported.")
                sys.exit(-1)
        return optimizer

    def get_regularizer(self, conf):
        regularizer = dict()
        for k, params in conf.items():
            try:
                regularizer[k] = EST.__dict__[k](**params)
            except ModuleNotFoundError:
                print(f"regularizer `{k}` is not supported.")
                sys.exit(-1)
        return regularizer
