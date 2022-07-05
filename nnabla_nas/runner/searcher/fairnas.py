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

from .search import Searcher
from ...utils.helper import SearchLogger
from ...utils.helper import ProgressMeter
from ...utils.estimator.memory import MemoryEstimator
import os
import nnabla as nn
import numpy as np


class FairNasSearcher(Searcher):
    r"""An implementation of FairNAS."""

    def __init__(self, model, optimizer, regularizer, dataloader, args):
        super().__init__(model, optimizer, regularizer, dataloader, args)
        # Number of models sampled at each batch
        self.m_sampled = args.get('num_sampled_iter', 4)
        # Number of samples for the search
        self.search_samples = args.get('num_search_samples', 0)
        self.logger = SearchLogger()
        self.search_monitor = ProgressMeter(
            self.search_samples,
            path=args['output_path'],
            quiet=self.comm.rank > 0,
            filename='log_search.txt')
        self.mest = MemoryEstimator()

        # loss and metric
        self.update_graph('valid')
        self.metrics = {
            k: nn.NdArray.from_numpy_array(np.zeros((1,)))
            for k in self.placeholder['valid']['metrics']
        }
        # loss and metric
        self.loss = nn.NdArray.from_numpy_array(np.zeros((1,)))

    def run(self):
        r"""Run the training process."""
        self.callback_on_start()
        self._start_warmup()

        # Training
        for self.cur_epoch in range(self.cur_epoch, self.args['epoch']):
            self.monitor.reset()
            lr = self.optimizer['train'].get_learning_rate()
            self.monitor.info(f'Running epoch={self.cur_epoch}\tlr={lr:.5f}\n')
            # training loop
            for i in range(self.one_epoch_train):
                self.train_on_batch()
                if i % (self.args['print_frequency']) == 0:
                    train_keys = [m.name for m in self.monitor.meters.values()
                                  if 'train' in m.name]
                    self.monitor.display(i, key=train_keys)
            # validation loop
            for i in range(len(self.dataloader['valid']) // self.bs_valid):
                # pick a random arch for each batch
                self.update_graph('valid')
                self.valid_on_batch()
            self.callback_on_epoch_end()
            self.monitor.write(self.cur_epoch)

        # Search
        for cur_sample in range(self.search_samples):
            self.search_monitor.reset()
            self.search_arch(sample_id=cur_sample)

        self.logger.save(self.args['output_path'])
        self.callback_on_finish()
        self.monitor.close()
        self.search_monitor.close()

    def callback_on_start(self):
        params_net = self.model.get_net_parameters(grad_only=True)
        self.optimizer['train'].set_parameters(params_net)

        # load checkpoint if available
        self.load_checkpoint()

        if self.comm.n_procs > 1:
            self._grads_net = [x.grad for x in params_net.values()]
            self.event.default_stream_synchronize()

    def train_on_batch(self):
        r"""Update the model parameters."""
        batch = [self.dataloader['train'].next()
                 for _ in range(self.accum_train)]
        bz, p = self.mbs_train, self.placeholder['train']
        self.optimizer['train'].zero_grad()

        if self.comm.n_procs > 1:
            self.event.default_stream_synchronize()

        # At each batch, accum gradient for m sampled models
        # then update params.
        for _ in range(self.m_sampled):
            self.update_graph('train')
            for data in batch:
                self._load_data(p, data)
                p['loss'].forward(clear_no_need_grad=True)
                for k, m in p['metrics'].items():
                    m.forward(clear_buffer=True)
                    self.monitor.update(f'{k}/train', m.d.copy(), bz)
                p['loss'].backward(clear_buffer=True)
                loss = p['loss'].d.copy()
                self.monitor.update('loss/train', loss * self.accum_train, bz)

        if self.comm.n_procs > 1:
            self.comm.all_reduce(self._grads_net, division=True, inplace=False)
            self.event.add_default_stream_event()

        self.optimizer['train'].update()

    def valid_on_batch(self, key='valid'):
        r"""validate an architecture from the search space"""
        bz, p = self.mbs_valid, self.placeholder['valid']

        if self.comm.n_procs > 1:
            self.event.default_stream_synchronize()
        for _ in range(self.accum_valid):
            self._load_data(p, self.dataloader[key].next())
            p['loss'].forward(clear_buffer=True)
            for k, m in p['metrics'].items():
                m.forward(clear_buffer=True)
                self.metrics[k].data += m.d.copy() * bz
            loss = p['loss'].d.copy()
            self.loss.data += loss * self.accum_valid * bz

        if self.comm.n_procs > 1:
            self.event.add_default_stream_event()
            self.comm.all_reduce(
                [self.loss] + list(self.metrics.values()), division=True, inplace=False)

    def callback_on_epoch_end(self):
        num_of_samples = self.one_epoch_valid * self.accum_valid * self.mbs_valid
        self.loss.data /= num_of_samples

        for k in self.metrics:
            self.metrics[k].data /= num_of_samples

        if self.comm.rank == 0:
            self.monitor.update('loss/valid', self.loss.data[0], 1)
            for k in self.metrics:
                self.monitor.update(f'{k}/valid', self.metrics[k].data[0], 1)
                self.monitor.info(f'{k}/valid {self.metrics[k].data[0]:.4f}\n')
            if self.args['save_nnp']:
                self.model.save_net_nnp(
                    self.args['output_path'],
                    self.placeholder['valid']['inputs'][0],
                    self.placeholder['valid']['outputs'][0],
                    save_params=self.args.get('save_params'))
            else:
                self.model.save_parameters(
                    path=os.path.join(self.args['output_path'], 'weights.h5')
                )
            # checkpoint
            self.save_checkpoint()
            if self.args['no_visualize']:  # action:store_false
                self.model.visualize(self.args['output_path'])

        # reset loss and metric
        self.loss.zero()
        for k in self.metrics:
            self.metrics[k].zero()

    def callback_on_finish(self):
        pass

    def search_arch(self, sample_id=0):
        r"""Validate an acrchitecture from the search space."""
        self.update_graph('valid')
        self.search_monitor.update(
            'search/n_parameters',
            self.mest.get_estimation(
                self.model))

        # Validation
        for i in range(len(self.dataloader['valid']) // self.bs_valid):
            self.valid_on_batch('valid')

        self.loss.data /= len(self.dataloader['valid'])
        for k in self.metrics:
            self.metrics[k].data /= len(self.dataloader['valid'])

        if self.comm.rank == 0:
            self.search_monitor.update('search/loss/valid', self.loss.data[0], 1)
            for k in self.metrics:
                self.search_monitor.update(
                    f'search/{k}/valid', self.metrics[k].data[0], 1)

        # Test
        for i in range(len(self.dataloader['test']) // self.bs_valid):
            self.valid_on_batch('test')

        self.loss.data /= len(self.dataloader['test'])
        for k in self.metrics:
            self.metrics[k].data /= len(self.dataloader['test'])

        if self.comm.rank == 0:
            self.search_monitor.update('search/loss/test', self.loss.data[0], 1)
            for k in self.metrics:
                self.search_monitor.update(
                    f'search/{k}/test', self.metrics[k].data[0], 1)

            self.logger.add_entry(sample_id,
                                  self.model.get_arch(),
                                  self.search_monitor.meters)
            self.logger.save(self.args['output_path'])
            self.search_monitor.write(sample_id)
            self.search_monitor.display(sample_id)

        # reset loss and metric
        self.loss.zero()
        for k in self.metrics:
            self.metrics[k].zero()
