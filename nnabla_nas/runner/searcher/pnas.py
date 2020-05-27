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

import nnabla as nn
import numpy as np

from .search import Searcher


class ProxylessNasSearcher(Searcher):
    r""" ProxylessNAS: Direct Neural Architecture Search on Target Task and
    Hardware.
    """

    def callback_on_start(self):
        r"""Gets the architecture parameters."""
        self._reward = nn.NdArray.from_numpy_array(np.zeros((1,)))

    def train_on_batch(self, key='train'):
        r"""Update the model parameters."""
        self.update_graph(key)
        params = self.model.get_net_parameters(grad_only=True)
        self.optimizer[key].set_parameters(params)
        bz, p = self.args.mbs_train, self.placeholder['train']
        self.optimizer[key].zero_grad()

        if self.comm.n_procs > 1:
            grads = [x.grad for x in params.values()]
            self.event.default_stream_synchronize()

        for _ in range(self.accum_train):
            self._load_data(p, self.dataloader['train'].next())
            p['loss'].forward(clear_no_need_grad=True)
            p['err'].forward(clear_buffer=True)
            p['loss'].backward(clear_buffer=True)
            loss, err = p['loss'].d.copy(), p['err'].d.copy()
            self.monitor.update('train_loss', loss * self.accum_train, bz)
            self.monitor.update('train_err', err, bz)

        if self.comm.n_procs > 1:
            self.comm.all_reduce(grads, division=True, inplace=False)
            self.event.add_default_stream_event()

        self.optimizer[key].update()

    def valid_on_batch(self):
        r"""Update the arch parameters."""
        beta, n_iter = 0.9, 10
        bz, p = self.args.mbs_valid, self.placeholder['valid']
        valid_data = [self.dataloader['valid'].next()
                      for i in range(self.accum_valid)]
        rewards, grads = [], []

        if self.comm.n_procs > 1:
            self.event.default_stream_synchronize()

        for _ in range(n_iter):
            reward = 0
            self.update_graph('valid')
            arch_params = self.model.get_arch_parameters(grad_only=True)
            self.optimizer['valid'].set_parameters(arch_params)
            for minibatch in valid_data:
                self._load_data(p, minibatch)
                p['loss'].forward(clear_buffer=True)
                p['err'].forward(clear_buffer=True)
                loss, err = p['loss'].d.copy(), p['err'].d.copy()
                reward += (1 - err) / self.accum_valid
                self.monitor.update('valid_loss', loss * self.accum_valid, bz)
                self.monitor.update('valid_err', err, bz)

            # adding constraints
            for k, v in self.regularizer.items():
                value = v['reg'].get_estimation(self.model)
                reward *= (min(1.0, v['bound'] / value))**v['weight']
                self.monitor.update(k, value, 1)
            rewards.append(reward)
            grads.append([m.g.copy() for m in arch_params.values()])

        # compute gradients
        for j, m in enumerate(arch_params.values()):
            m.grad.zero()
            for i, r in enumerate(rewards):
                m.g += (r - self._reward.data) * grads[i][j] / n_iter

        # update global reward
        self._reward.data = (beta * sum(rewards) / n_iter +
                             (1 - beta) * self._reward.data)

        if self.comm.n_procs > 1:
            self.comm.all_reduce(
                [x.grad for x in arch_params.values()],
                division=True,
                inplace=False
            )
            self.comm.all_reduce(self._reward, division=True, inplace=False)
            self.event.add_default_stream_event()

        self.monitor.update('reward', self._reward.data[0], self.args.bs_valid)
        self.optimizer['valid'].update()
