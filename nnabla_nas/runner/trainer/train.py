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

import nnabla as nn
import numpy as np
from tqdm import trange

from ...utils import helper
from ..runner import Runner


class Trainer(Runner):
    r"""Trainer class is a basic class for training a network."""

    def callback_on_start(self):
        r"""Builds the graphs and assigns parameters to the optimizers."""
        self.update_graph('train')
        params = self.model.get_parameters(grad_only=True)
        self.optimizer['train'].set_parameters(params)
        self.update_graph('valid')
        self._best_metric = {k: np.inf for k in self.placeholder['valid']['metrics']}

        # loss and metric
        self.loss = nn.NdArray.from_numpy_array(np.zeros((1,)))
        self.metrics = {
            k: nn.NdArray.from_numpy_array(np.zeros((1,)))
            for k in self.placeholder['valid']['metrics']
        }

        # calculate the model size
        model_size = helper.count_parameters(params)
        self.monitor.info('Model size = {:.6f} MB\n'.format(model_size*1e-6))

        # store a list of grads that will be synchronized
        if self.comm.n_procs > 1:
            self._grads = [x.grad for x in params.values()]

    def run(self):
        """Run the training process."""
        self.callback_on_start()

        for cur_epoch in range(self.args['epoch']):
            self.monitor.reset()
            lr = self.optimizer['train'].get_learning_rate()
            self.monitor.info(f'Running epoch={cur_epoch}\tlr={lr:.5f}\n')

            for i in range(self.one_epoch_train):
                self.train_on_batch()
                if i % (self.args['print_frequency']) == 0:
                    self.monitor.display(i, [k for k in self.monitor.meters if 'train' in k])

            for i in trange(self.one_epoch_valid, disable=self.comm.rank > 0):
                self.valid_on_batch()

            self.callback_on_epoch_end()
            self.monitor.write(cur_epoch)

        self.callback_on_finish()
        self.monitor.close()

    def train_on_batch(self, key='train'):
        r"""Updates the model parameters."""
        bz, p = self.mbs_train, self.placeholder['train']
        self.optimizer[key].zero_grad()

        if self.comm.n_procs > 1:
            self.event.default_stream_synchronize()

        for _ in range(self.accum_train):
            self._load_data(p, self.dataloader['train'].next())
            p['loss'].forward(clear_no_need_grad=True)
            for k, m in p['metrics'].items():
                m.forward(clear_buffer=True)
                self.monitor.update(f'{k}/train', m.d.copy(), bz)
            p['loss'].backward(clear_buffer=True)
            loss = p['loss'].d.copy()
            self.monitor.update('loss/train', loss * self.accum_train, bz)

        if self.comm.n_procs > 1:
            self.comm.all_reduce(self._grads, division=True, inplace=False)
            self.event.add_default_stream_event()

        self.optimizer[key].update()

    def valid_on_batch(self):
        r"""Runs the validation."""
        bz, p = self.mbs_valid, self.placeholder['valid']

        if self.comm.n_procs > 1:
            self.event.default_stream_synchronize()

        for _ in range(self.accum_valid):
            self._load_data(p, self.dataloader['valid'].next())
            p['loss'].forward(clear_buffer=True)
            for k, m in p['metrics'].items():
                m.forward(clear_buffer=True)
                self.metrics[k].data += m.d.copy() * bz
            loss = p['loss'].d.copy()
            self.loss.data += loss * self.accum_valid * bz

        if self.comm.n_procs > 1:
            self.event.add_default_stream_event()

    def callback_on_epoch_end(self):
        r"""Calculates the metric and saves the best parameters."""
        if self.comm.n_procs > 1:
            self.comm.all_reduce([self.loss]+list(self.metrics.values()), division=True, inplace=False)

        self.loss.data /= len(self.dataloader['valid'])
        for k in self.metrics:
            self.metrics[k].data /= len(self.dataloader['valid'])

        if self.comm.rank == 0:
            self.monitor.update('loss/valid', self.loss.data[0], 1)
            better = False
            for k in self.metrics:
                self.monitor.update(f'{k}/valid', self.metrics[k].data[0], 1)
                self.monitor.info(f'{k}={self.metrics[k].data[0]:.4f}\n')
                better |= self._best_metric[k] > self.metrics[k].data[0]
            if better:
                for k in self.metrics:
                    self._best_metric[k] = self.metrics[k].data[0]
                path = os.path.join(self.args['output_path'], 'weights.h5')
                self.model.save_parameters(path)

        # reset loss and metric
        self.loss.zero()
        for k in self.metrics:
            self.metrics[k].zero()

    def callback_on_finish(self):
        pass
