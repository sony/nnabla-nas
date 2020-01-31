import os
from collections import Counter, OrderedDict

import nnabla as nn
import nnabla.functions as F
import nnabla.utils.learning_rate_scheduler as LRS
from nnabla.logger import logger

from .. import utils as ut
from ..contrib.pnas import estimator as EST
from ..contrib.pnas.modules import CANDIDATE_FUNC, SampledOp
from ..dataset import DataLoader
from ..dataset.cifar10 import cifar10
from ..optimizer import Optimizer


class Searcher(object):
    """
    Searching the best architecture.
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
        for key in ['model', 'arch']:
            optim = conf[key + '_optimizer'].copy()
            lr_scheduler = ut.get_object_from_dict(
                module=LRS.__dict__,
                args=optim.pop('lr_scheduler', None)
            )
            solver = optim['solver']
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
        warmup = conf['warmup']
        one_epoch = len(self.loader['model']) // conf['batch_size']
        n_iter = 5
        # monitor the training process
        monitor = ut.ProgressMeter(one_epoch//n_iter, path=conf['output_path'])
        self._reward = 0  # average reward
        for cur_epoch in range(conf['epoch']):
            monitor.reset()
            for i in range(one_epoch//n_iter):
                for m in self.arch_modules:
                    m._update_active_idx()
                self._update_model_step(monitor, n_iter)
                if warmup == 0 and cur_epoch > 0:
                    self._update_arch_step(monitor)
                if i % (conf['print_frequency']//n_iter) == 0:
                    monitor.display(i)
            warmup -= warmup > 0
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

    def _update_model_step(self, monitor, n_iter=10):
        """Update the model parameters."""
        bz = self.conf['mini_batch_size']
        ph = self.placeholder['model']
        self._sample_train_net()
        for _ in range(n_iter):
            for _ in range(self.accum_grad):
                ph['input'].d, ph['target'].d = self.loader['model'].next()
                ph['loss'].forward(clear_no_need_grad=True)
                ph['loss'].backward(clear_buffer=True)
                ph['err'].forward(clear_buffer=True)
                monitor.update(
                    'model_loss', ph['loss'].d * self.accum_grad, bz)
                monitor.update('model_err', ph['err'].d, bz)
            self.optimizer['model'].update()

    def _update_arch_step(self, monitor):
        """Update the arch parameters."""
        beta, reward, bz = 0.9, 0, self.conf['mini_batch_size']
        ph = self.placeholder['arch']
        self._sample_search_net()
        for i in range(self.accum_grad):
            ph['input'].d, ph['target'].d = self.loader['arch'].next()
            ph['loss'].forward(clear_buffer=True)
            ph['err'].forward(clear_buffer=True)
            monitor.update('arch_loss', ph['loss'].d * self.accum_grad, bz)
            monitor.update('arch_err', ph['err'].d, bz)
            reward += (1 - ph['err'].d) / self.accum_grad
        # adding contraints
        for k, v in self.reg.items():
            value = v['reg'].get_estimation(self.model)
            reward *= (v['bound'] / value)**v['weight']
            monitor.update(k, value, 1)
        # compute gradients
        for j, m in enumerate(self.arch_modules):
            m._alpha.g *= reward - self._reward
        self.optimizer['arch'].update()
        # update average reward
        self._reward = beta * reward + (1 - beta) * self._reward
        monitor.update('reward', self._reward, self.conf['batch_size'])

    def _get_statistics(self):
        stats = ''
        ans = Counter([m._active for m in self.arch_modules])
        total = len(self.arch_modules)
        for k in range(len(self.op_names)):
            name = self.op_names[k]
            stats += name + f' = {ans[k]/total*100:.2f}%\t'
        return stats

    def _sample_train_net(self):
        """Sample a network for model update."""
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
        self.optimizer['model'].set_parameters(
            self.model.get_net_parameters(grad_only=True)
        )

    def _sample_search_net(self):
        """Sample a network for arch update."""
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
