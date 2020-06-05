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

from abc import ABC, abstractmethod

import numpy as np

from ..utils.helper import ProgressMeter


class Runner(ABC):
    r"""Runner is a basic class for training a model.

    You can adapt this class for your own runner by reimplementing the
    abstract methods of this class.

    Args:
        model (`nnabla_nas.contrib.model.Model`): The search model used to
            search the architecture.
        optimizer (dict): This stores optimizers for both `train` and `valid`
            graphs.
        dataloader (dict): This stores dataloaders for both `train` and `valid`
            graphs.
        args (Configuration): This stores all hyperparmeters used during
            training.
    """

    def __init__(self, model, optimizer, dataloader, args):

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.args = args

        # aditional argurments
        hp = self.args

        self.bs_train = hp['batch_size_train']
        self.mbs_train = hp['mini_batch_train']
        self.bs_valid = hp['batch_size_valid']
        self.mbs_valid = hp['mini_batch_valid']
        self.accum_train = self.bs_train // self.mbs_train
        self.accum_valid = self.bs_valid // self.mbs_valid
        self.one_epoch_train = len(self.dataloader['train']) // self.bs_train
        self.one_epoch_valid = len(self.dataloader['valid']) // self.bs_valid
        self.comm = hp['comm']
        self.event = hp['event']

        # monitor log info
        self.monitor = ProgressMeter(
            self.one_epoch_train,
            path=args.output_path,
            quiet=self.comm.rank > 0
        )

    @abstractmethod
    def run(self):
        r"""Run the training process."""
        pass

    def update_graph(self, key='train'):
        r"""Builds the graph and update the placeholder.

        Args:
            key (str, optional): Type of graph. Defaults to 'train'.
        """
        assert key in ('train', 'valid', 'warmup')

        self.model.apply(training=key != 'valid')
        p = self.placeholder['valid' if key == 'valid' else 'train']

        transform = self.transform['valid' if key == 'valid' else 'train']
        accum = self.accum_valid if key == 'valid' else self.accum_train

        # output features
        output, aux = self.model(transform(p['input'])), None
        if isinstance(output, tuple):
            aux, w = output[1], self.args.aux_weight
            p['output'] = output[0]
        else:
            p['output'] = output
        p['output'].apply(persistent=True)

        # loss function
        p['loss'] = self.criteria(p['output'], p['target']) / accum
        if aux is not None:
            p['loss'] += w * self.criteria(aux, p['target']) / accum
        p['loss'].apply(persistent=True)

        # top_n_error
        p['err'] = self.evaluate(
            p['output'].get_unlinked_variable().apply(need_grad=False),
            p['target']
        )
        p['err'].apply(persistent=True)

    @staticmethod
    def _load_data(placeholder, data):
        # TODO: improving the dataloader
        if isinstance(data[0], np.ndarray):
            placeholder['input'].d = data[0]
            placeholder['target'].d = data[1]
        else:
            placeholder['input'].data = data[0]
            placeholder['target'].data = data[1]

    @abstractmethod
    def train_on_batch(self, key='train'):
        r"""Runs the model update on a single batch of train data."""
        pass

    @abstractmethod
    def valid_on_batch(self):
        r"""Runs the model update on a single batch of valid data."""
        pass

    @abstractmethod
    def callback_on_epoch_end(self):
        r"""Calls this after one epoch."""
        pass

    @abstractmethod
    def callback_on_start(self):
        r"""Calls this on starting the run method."""
        pass

    @abstractmethod
    def callback_on_finish(self):
        r"""Calls this on finishing the run method."""
        pass
