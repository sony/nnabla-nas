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
import sys

from nnabla.logger import logger
import nnabla.utils.learning_rate_scheduler as LRS

from nnabla_nas import dataset
from nnabla_nas.optimizer import Optimizer
from nnabla_nas.utils import estimator as EST
from nnabla_nas.utils import helper


class Configuration(object):
    r"""A Configuration class base.

    Args:
        conf (dict): A dictionary containing configuration.
    """

    def __init__(self, conf):
        self.hparams = self.get_hyperparameters(conf['hparams'])
        self.dataloader = self.get_dataloader(conf['dataloader'])
        self.optimizer = self.get_optimizer(conf['optimizer'])

        # write the configuration
        path = self.hparams['output_path']
        Path(path).mkdir(parents=True, exist_ok=True)
        if self.hparams['comm'].rank == 0:
            file = os.path.join(path, 'config.json')
            logger.info(f'Saving the configurations to {file}')
            helper.write_to_json_file(conf, file)

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
            'searching': self.hparams['search'],
            'communicator': self.hparams['comm']
        })
        return {
            'train': loader_cls(training=True,
                                batch_size=self.hparams['mini_batch_train'], ** args),
            'valid': loader_cls(training=False,
                                batch_size=self.hparams['mini_batch_valid'], ** args)
        }

    def get_optimizer(self, conf):
        r"""Setup optimizer."""
        optimizer = dict()
        for name, args in conf.items():
            if name == 'regularizer':
                optimizer[name] = dict()
                for k, params in args.items():
                    try:
                        optimizer[name][k] = EST.__dict__[k](**params)
                    except ModuleNotFoundError:
                        print(f"regularizer `{k}` is not supported.")
                        sys.exit(-1)
            else:
                try:
                    lr_scheduler = None
                    if 'lr_scheduler' in args:
                        class_name = args['lr_scheduler']
                        lr = args['lr']
                        bz = self.hparams['batch_size_train'if name != 'valid' else 'batch_size_valid']
                        epoch = self.hparams['epoch'] if name == 'train' else self.hparams['warmup']
                        max_iter = epoch * len(self.dataloader['valid' if name == 'valid' else 'train']) // bz
                        lr_scheduler = LRS.__dict__[class_name](init_lr=lr, max_iter=max_iter)
                    args['lr_scheduler'] = lr_scheduler
                    optimizer[name] = Optimizer(**args)
                except ModuleNotFoundError:
                    print(f"optimizer `{name}` is not supported.")
                    sys.exit(-1)
        return optimizer
