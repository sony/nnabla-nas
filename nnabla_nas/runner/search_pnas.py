import os
from collections import Counter
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
import nnabla.utils.learning_rate_scheduler as LRS
from nnabla.logger import logger

from .. import utils as ut
from ..contrib.pnas import estimator as EST
from ..contrib.pnas.modules import CANDIDATE_FUNC
from ..contrib.pnas.modules import SampledOp
from ..dataset import DataLoader
from ..dataset.cifar10 import cifar10
from ..optimizer import Optimizer


class Searcher(object):
    """
    Searching the best architecture.

    Args:
        model ([type]): [description]
        conf ([type]): [description]

    """

    def __init__(self, model, conf):
        bz = conf['mini_batch_size']
        self.model = model
        self.conf = conf
        self.arch_modules = [m for _, m in model.get_modules()
                             if isinstance(m, SampledOp)]
        self.criteria = lambda o, t: F.mean(F.softmax_cross_entropy(o, t))
        self.evaluate = lambda o, t:  F.mean(F.top_n_error(o, t))
        self.accum_grad = self.conf['batch_size'] // bz
        self.op_names = list(CANDIDATE_FUNC.keys())

        # dataset configuration
        data = cifar10(conf['mini_batch_size'], True)
        train_transform, valid_transform = ut.dataset_transformer(conf)
        split = int(conf['train_portion'] * data.size)
        self.loader = {
            'model': DataLoader(
                data.slice(rng=None, slice_start=0, slice_end=split),
                train_transform
            ),
            'arch': DataLoader(
                data.slice(rng=None, slice_start=split, slice_end=data.size),
                valid_transform
            )
        }

        # regularizer configurations
        self.reg = dict()
        for k, v in conf['regularizer'].items():
            args = v.copy()
            self.reg[k] = dict()
            self.reg[k]['bound'] = args.pop('bound')
            self.reg[k]['weight'] = args.pop('weight')
            self.reg[k]['reg'] = EST.__dict__[args.pop('name')](**args)

        # solver configurations
        self.optimizer = dict()
        for key in ['model', 'arch', 'warmup']:
            optim = conf[key + '_optimizer'].copy()
            solver = optim['solver']
            lr_scheduler = None
            if key != 'arch':
                epoch = conf['epoch'] if key == 'model' else conf['warmup']
                lr_scheduler = LRS.__dict__['CosineScheduler'](
                    init_lr=solver['lr'],
                    max_iter=epoch * split // conf['batch_size']
                )
            self.optimizer[key] = Optimizer(
                retain_state=True,
                weight_decay=optim.pop('weight_decay', None),
                grad_clip=optim.pop('grad_clip', None),
                lr_scheduler=lr_scheduler,
                name=solver.pop('name'), **solver
            )

        # placeholders
        self.placeholder = OrderedDict({
            'model': {
                'input':  nn.Variable((bz, 3, 32, 32)),
                'target': nn.Variable((bz, 1))
            },
            'arch': {
                'input': nn.Variable((bz, 3, 32, 32)),
                'target': nn.Variable((bz, 1))
            }
        })

    def run(self):
        """Run the training process."""
        conf = self.conf
        one_epoch = len(self.loader['model']) // conf['batch_size']
        monitor = ut.ProgressMeter(one_epoch, path=conf['output_path'])

        # start with warm up
        self._warmup(monitor)

        self._reward = 0  # average reward
        for cur_epoch in range(conf['epoch']):
            monitor.reset()
            lr = self.optimizer['model'].get_learning_rate()
            logger.info(f'Running epoch={cur_epoch}\tlr={lr:.5f}')
            for i in range(one_epoch):
                self._update_model_step(monitor)
                self._update_arch_step(monitor)
                if i % (conf['print_frequency']) == 0:
                    monitor.display(i)

            # saving the architecture parameters
            nn.save_parameters(
                os.path.join(conf['output_path'], conf['model_name']) + '.h5',
                self.model.get_arch_parameters()
            )
            # logging output
            monitor.write(cur_epoch)

        logger.info(self._get_statistics())
        monitor.close()

        return self

    def _warmup(self, monitor):
        one_epoch = len(self.loader['model']) // self.conf['batch_size']
        for cur_epoch in range(self.conf['warmup']):
            monitor.reset()
            lr = self.optimizer['warmup'].get_learning_rate()
            logger.info(f'warm-up epoch={cur_epoch}\tlr={lr:.5f}')
            for i in range(one_epoch):
                self._update_model_step(monitor, key='warmup')
                if i % (self.conf['print_frequency']) == 0:
                    monitor.display(i)

    def _update_model_step(self, monitor, key='model'):
        """Update the model parameters."""
        bz = self.conf['mini_batch_size']
        ph = self.placeholder['model']
        self._sample_train_net(key=key)
        self.optimizer[key].zero_grad()
        for _ in range(self.accum_grad):
            ph['input'].d, ph['target'].d = self.loader['model'].next()
            ph['loss'].forward(clear_no_need_grad=True)
            ph['loss'].backward(clear_buffer=True)
            ph['err'].forward(clear_buffer=True)
            loss, err = ph['loss'].d.copy(),  ph['err'].d.copy()
            monitor.update('model_loss', loss * self.accum_grad, bz)
            monitor.update('model_err', err, bz)
        self.optimizer[key].update()

    def _update_arch_step(self, monitor):
        """Update the arch parameters."""
        beta, n_iter = 0.9, 5
        bz = self.conf['mini_batch_size']
        ph = self.placeholder['arch']
        data = [self.loader['arch'].next() for i in range(self.accum_grad)]
        rewards, grads = [], []
        for _ in range(n_iter):
            reward = 0
            self._sample_search_net()
            for i in range(self.accum_grad):
                ph['input'].d, ph['target'].d = data[i]
                ph['loss'].forward(clear_buffer=True)
                ph['err'].forward(clear_buffer=True)
                loss, err = ph['loss'].d.copy(),  ph['err'].d.copy()
                monitor.update('arch_loss', loss * self.accum_grad, bz)
                monitor.update('arch_err', err, bz)
                reward += (1 - err) / self.accum_grad
            # adding contraints
            for k, v in self.reg.items():
                value = v['reg'].get_estimation(self.model)
                reward *= (v['bound'] / value)**v['weight']
                monitor.update(k, value, 1)
            rewards.append(reward)
            grads.append([m._alpha.g.copy() for m in self.arch_modules])
            monitor.update('reward', reward, self.conf['batch_size'])
        # compute gradients
        for j, m in enumerate(self.arch_modules):
            m._alpha.grad.zero()
            for i, r in enumerate(rewards):
                m._alpha.g += (r - self._reward) * grads[i][j] / n_iter
        self.optimizer['arch'].update()
        self._reward = beta*sum(rewards)/n_iter + (1 - beta)*self._reward

    def _get_statistics(self):
        stats = ''
        ans = Counter([m._active for m in self.arch_modules])
        total = len(self.arch_modules)
        for k in range(len(self.op_names)):
            name = self.op_names[k]
            stats += name + f' = {ans[k]/total*100:.2f}%\t'
        return stats

    def _sample_train_net(self, key='model'):
        """Sample a network for model update."""
        for m in self.arch_modules:
            m._update_active_idx()
        self.model.apply(training=True)
        ph = self.placeholder['model']
        image = ut.image_augmentation(ph['input'])
        ph['output'] = self.model(image).apply(persistent=True)
        ph['loss'] = self.criteria(ph['output'], ph['target'])/self.accum_grad
        ph['err'] = self.evaluate(
            ph['output'].get_unlinked_variable(),
            ph['target']
        )
        ph['loss'].apply(persistent=True)
        ph['err'].apply(persistent=True)
        # setup parameters
        self.optimizer[key].set_parameters(
            self.model.get_net_parameters(grad_only=True)
        )

    def _sample_search_net(self):
        """Sample a network for arch update."""
        for m in self.arch_modules:
            m._update_active_idx()
        self.model.apply(training=False)
        ph = self.placeholder['arch']
        ph['output'] = self.model(ph['input']).apply(persistent=True)
        ph['loss'] = self.criteria(ph['output'], ph['target'])/self.accum_grad
        ph['err'] = self.evaluate(
            ph['output'].get_unlinked_variable(),
            ph['target']
        )
        ph['loss'].apply(persistent=True)
        ph['err'].apply(persistent=True)
        self.optimizer['arch'].set_parameters(
            self.model.get_arch_parameters(grad_only=True)
        )
