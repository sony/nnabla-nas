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

from abc import ABC
from abc import abstractmethod

import nnabla as nn

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

        # setup placeholder
        self.placeholder = {
            'train': {
                'inputs': [nn.Variable([self.mbs_train] + shape) for shape in args['input_shapes']],
                'targets': [nn.Variable([self.mbs_train] + shape) for shape in args['target_shapes']]
            },
            'valid': {
                'inputs': [nn.Variable([self.mbs_valid] + shape) for shape in args['input_shapes']],
                'targets': [nn.Variable([self.mbs_valid] + shape) for shape in args['target_shapes']]
            }
        }

        # monitor log info
        self.monitor = ProgressMeter(self.one_epoch_train, path=args['output_path'], quiet=self.comm.rank > 0)

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

        fake_key = 'valid' if key == 'valid' else 'train'
        p = self.placeholder[fake_key]
        transform = self.dataloader[fake_key].transform(fake_key)
        accum = self.accum_valid if key == 'valid' else self.accum_train

        # outputs
        inputs = [transform(x) for x in p['inputs']]
        outputs = self.model(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)

        p['outputs'] = [x.apply(persistent=True) for x in outputs]

        # loss function
        p['loss'] = self.model.loss(p['outputs'], p['targets'], self.args['loss_weights']) / accum
        p['loss'].apply(persistent=True)

        # metrics to monitor during training
        targets = [out.get_unlinked_variable().apply(need_grad=False) for out in p['outputs']]
        p['metrics'] = self.model.metrics(targets, p['targets'])
        for v in p['metrics'].values():
            v.apply(persistent=True)

    @staticmethod
    def _load_data(placeholder, data):
        for key in ('inputs', 'targets'):
            for inp, x in zip(placeholder[key], data[key]):
                if isinstance(x, nn.NdArray):
                    inp.data = x
                else:
                    inp.d = x

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
