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


class DartsSearcher(Searcher):
    r"""An implementation of DARTS: Differentiable Architecture Search."""

    def callback_on_start(self):
        r"""Builds the graphs and assigns parameters to the optimizers."""
        self.update_graph('train')
        params_net = self.model.get_net_parameters(grad_only=True)
        self.optimizer['train'].set_parameters(params_net)

        self.update_graph('valid')
        params_arch = self.model.get_arch_parameters(grad_only=True)
        self.optimizer['valid'].set_parameters(params_arch)

        if self.comm.n_procs > 1:
            self._grads_net = [x.grad for x in params_net.values()]
            self._grads_arch = [x.grad for x in params_arch.values()]
            self.event.default_stream_synchronize()

    def train_on_batch(self, key='train'):
        r"""Updates the model parameters."""
        bz, p = self.args.mbs_train, self.placeholder['train']
        self.optimizer[key].zero_grad()

        if self.comm.n_procs > 1:
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
            self.comm.all_reduce(self._grads_net, division=True, inplace=False)
            self.event.add_default_stream_event()

        self.optimizer[key].update()

    def valid_on_batch(self):
        r"""Updates the architecture parameters."""
        bz, p = self.args.mbs_valid, self.placeholder['valid']
        self.optimizer['valid'].zero_grad()

        if self.comm.n_procs > 1:
            self.event.default_stream_synchronize()

        for _ in range(self.accum_valid):
            self._load_data(p, self.dataloader['valid'].next())
            p['loss'].forward(clear_no_need_grad=True)
            p['err'].forward(clear_buffer=True)
            p['loss'].backward(clear_buffer=True)
            loss, err = p['loss'].d.copy(),  p['err'].d.copy()
            self.monitor.update('valid_loss', loss * self.accum_valid, bz)
            self.monitor.update('valid_err', err, bz)

        if self.comm.n_procs > 1:
            self.comm.all_reduce(
                self._grads_arch, division=True, inplace=False)
            self.event.add_default_stream_event()

        self.optimizer['valid'].update()
