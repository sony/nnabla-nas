import os

import nnabla as nn

from ...contrib.darts.modules import MixedOp
from .search import Searcher


class ProxylessNasSearcher(Searcher):
    r""" ProxylessNAS: Direct Neural Architecture Search on Target Task and
    Hardware.
    """

    def callback_on_start(self):
        r"""Gets the architecture modules."""
        self.arch_modules = [
            m for _, m in self.model.get_modules()
            if isinstance(m, MixedOp)
        ]
        self._reward = 0

    def callback_on_sample_graph(self):
        r"""Samples a new graph."""
        for m in self.arch_modules:
            m.update_active_index()

    def train_on_batch(self, key='train'):
        r"""Update the model parameters."""
        self.update_graph(key)
        self.optimizer[key].set_parameters(
            self.model.get_net_parameters(grad_only=True)
        )
        bz, p = self.args.mbs_train, self.placeholder['train']
        self.optimizer[key].zero_grad()
        for _ in range(self.accum_train):
            p['input'].d, p['target'].d = self.dataloader['train'].next()
            p['loss'].forward(clear_no_need_grad=True)
            p['loss'].backward(clear_buffer=True)
            p['err'].forward(clear_buffer=True)
            loss, err = p['loss'].d.copy(),  p['err'].d.copy()
            self.monitor.update('train_loss', loss * self.accum_train, bz)
            self.monitor.update('train_err', err, bz)
        self.optimizer[key].update()

    def valid_on_batch(self):
        r"""Update the arch parameters."""
        beta, n_iter = 0.9, 5
        bz, p = self.args.mbs_valid, self.placeholder['valid']
        valid_data = [self.dataloader['valid'].next()
                      for i in range(self.accum_valid)]
        rewards, grads = [], []
        for _ in range(n_iter):
            reward = 0
            self.update_graph('valid')
            self.optimizer['valid'].set_parameters(
                self.model.get_arch_parameters(grad_only=True)
            )
            for minibatch in valid_data:
                p['input'].d, p['target'].d = minibatch
                p['loss'].forward(clear_buffer=True)
                p['err'].forward(clear_buffer=True)
                loss, err = p['loss'].d.copy(), p['err'].d.copy()
                reward += (1 - err) / self.accum_valid
                self.monitor.update('valid_loss', loss * self.accum_valid, bz)
                self.monitor.update('valid_err', err, bz)
            # adding contraints
            for k, v in self.regularizer.items():
                value = v['reg'].get_estimation(self.model)
                reward *= (min(1.0, v['bound'] / value))**v['weight']
                self.monitor.update(k, value, 1)
            rewards.append(reward)
            grads.append([m._alpha.g.copy() for m in self.arch_modules])
            self.monitor.update('reward', reward, self.args.bs_valid)
        # compute gradients
        for j, m in enumerate(self.arch_modules):
            m._alpha.grad.zero()
            for i, r in enumerate(rewards):
                m._alpha.g += (r - self._reward) * grads[i][j] / n_iter
        self.optimizer['valid'].update()
        self._reward = beta*sum(rewards)/n_iter + (1 - beta)*self._reward

    def callback_on_epoch_end(self):
        r"""Calls this after one epoch."""
        nn.save_parameters(
            os.path.join(self.args.output_path, 'arch.h5'),
            self.model.get_parameters()
        )
        self.monitor.info(self.model.summary() + '\n')
