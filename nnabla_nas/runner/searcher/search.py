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

from ..runner import Runner


class Searcher(Runner):
    r"""Searching the best architecture."""

    def run(self):
        r"""Run the training process."""
        self.callback_on_start()
        if self.cur_epoch == 0:
            # do not run warmup if start from checkpoint
            self._start_warmup()

        for self.cur_epoch in range(self.cur_epoch, self.hparams['epoch']):
            self.monitor.reset()
            lr = self.optimizer['train'].get_learning_rate()
            self.monitor.info(f'Running epoch={self.cur_epoch}\tlr={lr:.5f}\n')

            for i in range(self.one_epoch_train):
                self.train_on_batch()
                self.valid_on_batch()
                if i % (self.args['print_frequency']) == 0:
                    self.monitor.display(i)

            self.callback_on_epoch_end()
            self.monitor.write(self.cur_epoch)

        self.callback_on_finish()
        self.monitor.close()
        return self

    def _start_warmup(self):
        r"""Performs warmup for the model on training."""
        for cur_epoch in range(self.hparams['warmup']):
            self.monitor.reset()

            lr = self.optimizer['warmup'].get_learning_rate()
            self.monitor.info(f'warm-up epoch={cur_epoch}\tlr={lr:.5f}\n')

            for i in range(self.one_epoch_train):
                self.train_on_batch(key='warmup')
                if i % (self.args['print_frequency']) == 0:
                    self.monitor.display(i)

    def callback_on_epoch_end(self):
        r"""Calls this after one epoch."""
        if self.comm.rank == 0:
            if self.args['save_nnp']:
                self.model.save_net_nnp(
                    self._abs_output_path,
                    self.placeholder['valid']['inputs'][0],
                    self.placeholder['valid']['outputs'][0],
                    save_params=self.args.get('save_params'))
            else:
                self.model.save_parameters(
                    path=os.path.join(self._abs_output_path, 'arch.h5'),
                    params=self.model.get_arch_parameters()
                )
            # checkpoint
            self.save_checkpoint()
            if self.args['no_visualize']:  # action:store_false
                self.model.visualize(self._abs_output_path)

        self.monitor.info(self.model.summary() + '\n')

    def callback_on_finish(self):
        r"""Calls this on finishing the training."""
        if self.comm.rank == 0:
            if self.args['save_nnp']:
                self.model.save_net_nnp(
                    self._abs_output_path,
                    self.placeholder['valid']['inputs'][0],
                    self.placeholder['valid']['outputs'][0],
                    save_params=self.args.get('save_params'))
            else:
                self.model.save_parameters(
                    path=os.path.join(self._abs_output_path, 'weights.h5'),
                    params=self.model.get_net_parameters()
                )
            if self.args['no_visualize']:  # action:store_false
                self.model.visualize(self._abs_output_path)

    def callback_on_start(self):
        r"""Calls this on starting the training."""
        # load checkpoint if available
        self.load_checkpoint()
