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
from more_itertools import consume
from pathlib import Path
from os import environ
import json

import nnabla as nn

from ..utils.helper import ProgressMeter, get_output_path
from ..utils.data import transforms


class Runner(ABC):
    r"""Runner is a basic class for training a model.

    You can adapt this class for your own runner by reimplementing the
    abstract methods of this class.

    Args:
        model (`nnabla_nas.contrib.model.Model`): The search model used to
            search the architecture.
        optimizer (dict): This stores optimizers for both `train` and `valid`
            graphs. Must only store instances of `Optinmizer`
        regularizer (dict): This stores regularizers such as the latency and memory
            estimators
        dataloader (dict): This stores dataloaders for both `train` and `valid`
            graphs.
        hparams  (Configuration): This stores all hyperparmeters used during training.
        args (Configuration): This stores other variables used during for training:
             event, communicator, output_path...
    """

    def __init__(self, model, optimizer, regularizer, dataloader, hparams, args):

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.hparams = hparams
        self.args = args

        # aditional argurments
        hp = self.hparams
        self.comm = args['comm']
        self.event = args['event']
        n_procs = self.comm.n_procs

        # hp['batch_size_XX'] is the GLOBAL batch size used for train / val
        #     which is independent on the number of GPUs
        # self.bs_XX is the LOCAL batch size (the batch size for each used GPU)
        # self.mbs_XX is the MINIBATCH size, the nr. of samples used at once
        # for forward pass of the network
        self.bs_train = hp['batch_size_train'] // n_procs
        self.mbs_train = hp['mini_batch_train']
        self.bs_valid = hp['batch_size_valid'] // n_procs
        self.mbs_valid = hp['mini_batch_valid']
        self.accum_train = self.bs_train // self.mbs_train
        self.accum_valid = self.bs_valid // self.mbs_valid
        self.one_epoch_train = len(self.dataloader['train']) // self.bs_train
        self.one_epoch_valid = len(self.dataloader['valid']) // self.bs_valid
        self.cur_epoch = 0

        # setup placeholder
        def create_variables(mbs, shapes):
            return [nn.Variable([mbs] + shape) for shape in shapes]

        self.placeholder = {}
        self.placeholder['train'] = {
            'inputs': create_variables(self.mbs_train, hparams['input_shapes']),
            'targets': create_variables(self.mbs_train, hparams['target_shapes'])
        }
        self.placeholder['valid'] = {
            'inputs': create_variables(self.mbs_valid, hparams['input_shapes']),
            'targets': create_variables(self.mbs_valid, hparams['target_shapes'])
        }

        # monitor log info
        self._abs_output_path = str(Path(get_output_path(is_abspath=True)) / self.args['output_path'])
        self._rel_output_path = str(Path(get_output_path(is_abspath=False)) / self.args['output_path'])
        self.monitor = ProgressMeter(self.one_epoch_train, self._abs_output_path,
                                     quiet=self.comm.rank > 0)

        # Check if we should run in fast mode where all computation graph is
        # kept in memory and mixed operations just switch the propagation path.
        fast_mode = environ.get('NNABLA_NAS_MIXEDOP_FAST_MODE') is not None
        self.monitor.info('NNABLA_NAS_MIXEDOP_FAST_MODE is {}\n'.format(
                          'enabled' if fast_mode else 'disabled'))
        self._fast_mode = fast_mode

    @property
    def fast_mode(self):
        return self._fast_mode

    @abstractmethod
    def run(self):
        r"""Run the training process."""
        pass

    def update_graph(self, key='train'):
        r"""Builds the graph and update the placeholder.

        Args:
            key (str, optional): Type of graph. Defaults to 'train'.
        """
        if key not in ('train', 'valid', 'warmup'):
            raise ValueError(f'key = {key} is not allowed')

        if key in ('train', 'warmup'):
            key = 'train'

        placeholder = self.placeholder[key]

        if self.dataloader[key].transform is None:
            self.dataloader[key].transform = 'none_transform'

        try:
            # self.dataloader[key].transform is a str with the name of the tranformation to apply
            func = getattr(transforms, self.dataloader[key].transform)
            # transform is the corresponding function in utils/data/transforms.py to be used
            transform = func(key)
        except AttributeError:
            print(
                'ERROR: Transformation function \'' +
                self.dataloader[key].transform +
                '\' NOT defined in ' + transforms.__name__
                )
            raise AttributeError

        training = key == 'train'
        model = self.model

        # apply data transformations
        if not self.fast_mode or 'transformed' not in placeholder:
            inputs = placeholder['inputs']
            inputs = [transform(x) for x in inputs]
            placeholder['transformed'] = inputs

        inputs = placeholder['transformed']

        # generate a new architecture
        model.apply(None, training=training)
        outputs = model(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        outputs = [output.apply(persistent=True) for output in outputs]
        placeholder['outputs'] = outputs

        # add the model's loss function
        if not self.fast_mode or 'loss' not in placeholder:
            targets = placeholder['targets']
            loss_weights = self.hparams['loss_weights']
            accum = self.accum_train if training else self.accum_valid
            loss = model.loss(outputs, targets, loss_weights) / accum
            placeholder['loss'] = loss.apply(persistent=True)

        # metrics to monitor during training
        if not self.fast_mode or 'metrics' not in placeholder:
            targets = placeholder['targets']
            outputs = (v.get_unlinked_variable() for v in outputs)
            outputs = list(v.apply(need_grad=False) for v in outputs)
            metrics = model.metrics(outputs, targets)
            consume(v.apply(persistent=True) for v in metrics.values())
            placeholder['metrics'] = metrics

    @staticmethod
    def _load_data(placeholder, data):
        for key in ('inputs', 'targets'):
            for inp, x in zip(placeholder[key], data[key]):
                if isinstance(x, nn.NdArray):
                    inp.data = x
                else:
                    inp.d = x

    def save_checkpoint(self, checkpoint_info={}):
        r"""Save the current states of the runner."""
        path = Path(self._abs_output_path) / 'checkpoint'
        path.mkdir(parents=True, exist_ok=True)
        relpath = Path(self._rel_output_path) / 'checkpoint'

        checkpoint_info['epoch'] = self.cur_epoch

        # save optimizers state
        checkpoint_info['optimizers'] = dict()
        for name, optimizer in self.optimizer.items():
            checkpoint_info['optimizers'][name] = optimizer.save_checkpoint(str(relpath), name)

        if ("best_metric" in checkpoint_info.keys() and "error" in checkpoint_info["best_metric"].keys()):
            checkpoint_info["best_metric"]["error"] = float(checkpoint_info["best_metric"]["error"])

        # save parameters
        self.model.save_parameters(str(path / 'weights.h5'))
        checkpoint_info['params_path'] = str(relpath / 'weights.h5')

        with path.joinpath('checkpoint.json').open('w') as f:
            json.dump(checkpoint_info, f)

        self.monitor.info(f"Checkpoint saved: {str(path)}\n")

    def load_checkpoint(self):

        output_path = get_output_path()

        path = Path(output_path) / 'checkpoint' / 'checkpoint.json'
        if path.is_file():
            # path = os.path.join(path, 'checkpoint.json')
            with path.open('r') as f:
                checkpoint_info = json.load(f)

            self.cur_epoch = checkpoint_info['epoch'] + 1
            # load optimizers
            for name, optim_info in checkpoint_info['optimizers'].items():
                p = self.model.get_parameters()
                # make sure that optimizer parameters match
                params_names = checkpoint_info['optimizers'][name]['params_names']
                params = {k: p[k] for k in params_names}
                self.optimizer[name].set_parameters(params)
                self.optimizer[name].load_checkpoint(optim_info)

            # load parameters
            self.model.load_parameters(checkpoint_info['params_path'])
            self.monitor.info(f"Checkpoint loaded: {str(path)}\n")
            return checkpoint_info
        return None

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
