from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import nnabla as nn
from nnabla.logger import logger

from .. import utils as ut
from ..utils import ProgressMeter


class Searcher(object):
    r"""Searching the best architecture.

    Args:
        model (``nnabla_nas.): [description]
        placeholder ([type]): [description]
        optimizer ([type]): [description]
        dataloader ([type]): [description]
        regularizer ([type]): [description]
        criteria ([type]): [description]
        evaluate ([type]): [description]
        args ([type]): [description]
    """

    def __init__(self, model,  placeholder, optimizer, dataloader, regularizer,
                 criteria, evaluate, args):
        self.model = model
        self.criteria = criteria
        self.evaluate = evaluate
        self.dataloader = dataloader
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.placeholder = placeholder
        self.args = args

        # aditional argurments
        self.accum_train = self.args.bs_train // self.args.mbs_train
        self.accum_valid = self.args.bs_valid // self.args.mbs_valid
        self.one_epoch = len(self.dataloader['train']) // args.bs_train
        self.monitor = ProgressMeter(self.one_epoch, path=args.output_path)

        self.callback_on_start()

    def run(self):
        """Run the training process."""
        self._warmup()
        for cur_epoch in range(self.args.epoch):
            self.monitor.reset()
            lr = self.optimizer['train'].get_learning_rate()
            logger.info(f'Running epoch={cur_epoch}\tlr={lr:.5f}')
            for i in range(self.one_epoch):
                self.callback_on_update_model()
                self.callback_on_update_arch()
                if i % (self.args.print_frequency) == 0:
                    self.monitor.display(i)
            self.callback_on_update()
            self.monitor.write(cur_epoch)
        self.callback_on_finish()
        self.monitor.close()
        return self

    def _warmup(self):
        """Performs warmup for the model on training."""
        for cur_epoch in range(self.args.warmup):
            self.monitor.reset()
            lr = self.optimizer['warmup'].get_learning_rate()
            logger.info(f'warm-up epoch={cur_epoch}\tlr={lr:.5f}')
            for i in range(self.one_epoch):
                self.callback_on_update_model(key='warmup')
                if i % (self.args.print_frequency) == 0:
                    self.monitor.display(i)

    def update_graph(self, key='train'):
        r"""Builds the graph and assigns parameters to the optimizer.

        Args:
            key (str, optional): Type of graph. Defaults to 'train'.
        """
        self.callback_on_sample_network()
        self.model.apply(training=key != 'valid')
        p = self.placeholder['valid' if key == 'valid' else 'train']
        image = (ut.image_augmentation(p['input']) if key == 'valid'
                 else p['input'])
        accum = self.accum_valid if key == 'valid' else self.accum_train
        p['output'] = self.model(image).apply(persistent=True)
        p['loss'] = self.criteria(p['output'], p['target']) / accum
        p['err'] = self.evaluate(
            p['output'].get_unlinked_variable(),
            p['target']
        )
        p['loss'].apply(persistent=True)
        p['err'].apply(persistent=True)
        # setup parameters
        self.optimizer[key].set_parameters(
            self.model.get_arch_parameters(grad_only=True) if key == 'valid'
            else self.model.get_net_parameters(grad_only=True)
        )

    def callback_on_update(self):
        r"""Calls this after one epoch."""
        # save the model parameters
        nn.save_parameters(
            os.path.join(self.args.output_path, 'arch.h5'),
            self.model.get_parameters()
        )

    def callback_on_sample_network(self):
        r"""Calls this before sample a graph."""
        pass

    def callback_on_finish(self):
        r"""Calls this after finishing the training."""
        pass

    def callback_on_start(self):
        r"""Calls this on starting."""
        pass

    def callback_on_update_model(self):
        r"""Calls this when updating the model parameters."""
        raise NotImplementedError

    def callback_on_update_arch(self):
        r"""Calls this when updating the arch parameters."""
        raise NotImplementedError


class DartsSeacher(Searcher):

    def callback_on_start(self):
        self.update_graph('train')
        self.update_graph('valid')

    def callback_on_update_model(self, key='train'):
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

    def callback_on_update_arch(self):
        bz, p = self.args.mbs_valid, self.placeholder['valid']
        self.optimizer['valid'].zero_grad()
        for i in range(self.accum_valid):
            p['input'].d, p['target'].d = self.dataloader['valid'].next()
            p['loss'].forward(clear_buffer=True)
            p['err'].forward(clear_buffer=True)
            loss, err = p['loss'].d.copy(),  p['err'].d.copy()
            self.monitor.update('valid_loss', loss * self.accum_valid, bz)
            self.monitor.update('valid_err', err, bz)
        self.optimizer['valid'].update()


class ProxylessNasSearcher(Searcher):

    def callback_on_start(self):
        self.arch_modules = self.model.get_arch_modules()
        self._reward = 0

    def callback_on_sample_network(self):
        for m in self.arch_modules:
            m.update_active_index()

    def callback_on_update_model(self, key='train'):
        r"""Update the model parameters."""
        self.update_graph(key)
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

    def callback_on_update_arch(self):
        r"""Update the arch parameters."""
        beta, n_iter = 0.9, 5
        bz, p = self.args.mbs_valid, self.placeholder['valid']
        data = [self.dataloader['valid'].next()
                for i in range(self.accum_valid)]
        rewards, grads = [], []
        for _ in range(n_iter):
            reward = 0
            self.update_graph('valid')
            for i in range(self.accum_valid):
                p['input'].d, p['target'].d = data[i]
                p['loss'].forward(clear_buffer=True)
                p['err'].forward(clear_buffer=True)
                loss, err = p['loss'].d.copy(),   p['err'].d.copy()
                reward += (1 - err) / self.accum_valid
                self.monitor.update('valid_loss', loss * self.accum_valid, bz)
                self.monitor.update('valid_err', err, bz)
            # adding contraints
            for k, v in self.regularizer.items():
                value = v['reg'].get_estimation(self.model)
                reward *= (v['bound'] / value)**v['weight']
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
