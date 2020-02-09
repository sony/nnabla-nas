import os

import nnabla as nn
import numpy as np

from ... import utils as ut
from ...contrib.darts.modules import CANDIDATES
from .search import Searcher


class DartsSearcher(Searcher):
    r"""An implementation of DARTS: Differentiable Architecture Search."""

    def callback_on_start(self):
        r"""Builds the graphs and assigns parameters to the optimizers."""
        self.update_graph('train')
        self.optimizer['train'].set_parameters(
            self.model.get_net_parameters(grad_only=True)
        )
        self.update_graph('valid')
        self.optimizer['valid'].set_parameters(
            self.model.get_arch_parameters(grad_only=True)
        )

    def train_on_batch(self, key='train'):
        r"""Updates the model parameters."""
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
        r"""Updates the architecture parameters."""
        bz, p = self.args.mbs_valid, self.placeholder['valid']
        self.optimizer['valid'].zero_grad()
        for _ in range(self.accum_valid):
            p['input'].d, p['target'].d = self.dataloader['valid'].next()
            p['loss'].forward(clear_no_need_grad=True)
            p['loss'].backward(clear_buffer=True)
            p['err'].forward(clear_buffer=True)
            loss, err = p['loss'].d.copy(),  p['err'].d.copy()
            self.monitor.update('valid_loss', loss * self.accum_valid, bz)
            self.monitor.update('valid_err', err, bz)
        self.optimizer['valid'].update()

    def callback_on_epoch_end(self):
        r"""Saves parameters and prints the selection propability."""
        nn.save_parameters(
            os.path.join(self.args.output_path, 'arch.h5'),
            self.model.get_parameters()
        )
        op_names = list(CANDIDATES.keys())
        for alphas, t in zip(self.model._alpha, ['normal', 'reduction']):
            count = {i: 0 for i in op_names}
            for alpha in alphas:
                idx = np.argmax(alpha.d.flat)
                count[op_names[idx]] += 1
            select = t + ' cell:\n'
            for k, v in count.items():
                select += f'{k} = {v/len(alphas)*100:.2f}%\t'
            self.monitor.info(select + '\n')

    def callback_on_finish(self):
        r"""Visualize the architecture for darts."""
        ut.save_dart_arch(self.model, self.args.output_path)
