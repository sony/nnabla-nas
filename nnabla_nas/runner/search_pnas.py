import os
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
import nnabla.utils.learning_rate_scheduler as LRS
from nnabla.logger import logger
from collections import Counter
import numpy as np
from scipy.special import softmax

from ..contrib.darts.modules import CANDIDATE_FUNC
from .. import utils as ut
from ..dataset import DataLoader
from ..dataset.cifar10 import cifar10
from ..optimizer import Optimizer
from ..contrib.pnas import estimator as EST


class Searcher(object):
    """
    Searching the best architecture.
    """

    def __init__(self, model, conf):
        self.model = model
        self.conf = conf
        self.arch_modules = model.get_arch_modules()
        self.criteria = lambda o, t: F.mean(F.softmax_cross_entropy(o, t))
        self.evaluate = lambda o, t:  F.mean(F.top_n_error(o, t))
        self.iter = self.conf['batch_size'] // self.conf['mini_batch_size']
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
        self.reg = None
        if 'regularizer' in conf:
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
                retain_state=key == 'model',
                weight_decay=optim.pop('weight_decay', None),
                grad_clip=optim.pop('grad_clip', None),
                lr_scheduler=lr_scheduler,
                name=solver.pop('name'), **solver
            )

        # placeholders
        self.placeholder = OrderedDict({
            'model': {
                'input':  nn.Variable((conf['mini_batch_size'], 3, 32, 32)),
                'target': nn.Variable((conf['mini_batch_size'], 1))
            },
            'arch': {
                'input': nn.Variable((conf['mini_batch_size'], 3, 32, 32)),
                'target': nn.Variable((conf['mini_batch_size'], 1))
            }
        })

    def run(self):
        """Run the training process."""

        conf = self.conf
        warmup = conf['warmup']
        one_epoch = len(self.loader['model']) // conf['batch_size']

        # monitor the training process
        monitor = ut.ProgressMeter(one_epoch, path=conf['output_path'])
        ut.write_to_json_file(
            content=conf,
            file_path=os.path.join(conf['output_path'], 'search_config.json')
        )

        self._avg_reward = None  # average reward
        for cur_epoch in range(conf['epoch']):
            monitor.reset()
            for i in range(one_epoch):
                self._update_model_step(monitor)
                if warmup == 0:
                    self._update_arch_step(monitor)
                if i % conf['print_frequency'] == 0:
                    monitor.display(i)
            warmup -= warmup > 0
            # saving the architecture parameters
            nn.save_parameters(
                os.path.join(conf['output_path'], conf['model_name']) + '.h5',
                self.model.get_arch_parameters()
            )
            # logging output
            logger.info(self._get_statistics())
            monitor.write(cur_epoch)

        monitor.close()
        return self

    def _update_model_step(self, monitor):
        """Update the model parameters."""
        bz = self.conf['mini_batch_size']
        ph = self.placeholder['model']
        self._sample_train_net()
        for _ in range(self.iter):
            ph['input'].d, ph['target'].d = self.loader['model'].next()
            ph['loss'].forward(clear_no_need_grad=True)
            ph['err'].forward(clear_buffer=True)
            ph['loss'].backward(clear_buffer=True)
            monitor.update('model_loss', ph['loss'].d * self.iter, bz)
            monitor.update('model_err', ph['err'].d, bz)
        self.optimizer['model'].update()

    def _update_arch_step(self, monitor):
        """Update the arch parameters."""
        n_iter = 10
        # save temporal values
        reward_list, grad_list = [], []
        # sample a minibatch
        vd, vt = [None]*self.iter, [None]*self.iter
        for i in range(self.iter):
            vd[i], vt[i] = self.loader['arch'].next()
        bz = self.conf['mini_batch_size']
        ph = self.placeholder['arch']

        for _ in range(n_iter):
            reward = 0
            self._sample_search_net()
            for i in range(self.iter):
                ph['input'].d, ph['target'].d = vd[i], vt[i]
                ph['loss'].forward(clear_no_need_grad=True)
                ph['err'].forward(clear_buffer=True)
                monitor.update('arch_loss', ph['loss'].d * self.iter, bz)
                monitor.update('arch_err', ph['err'].d, bz)
                reward += 1 - ph['err'].d

            reward /= self.iter
            for k, v in self.reg.items():
                value = v['reg'].get_estimation(self.model)
                reward *= (v['bound'] / value)**v['weight']
                monitor.update(k, value, 1)

            reward_list.append(reward)

            grad_cur = []
            for m in self.arch_modules:
                probs = softmax(m._alpha.d.flat)
                probs[m._active] -= 1
                grad_cur.append(probs)

            grad_list.append(grad_cur)

        avg_reward = sum(reward_list) / n_iter

        if self._avg_reward is None:
            self._avg_reward = avg_reward
        else:
            self._avg_reward += 0.99 * (avg_reward - self._avg_reward)

        for j, m in enumerate(self.arch_modules):
            probs = np.zeros(m._alpha.shape).flatten()
            m._alpha.grad.zero()
            for i in range(n_iter):
                probs += (reward_list[i] - self._avg_reward) * \
                    grad_list[i][j]
            m._alpha.g = probs.reshape(m._alpha.shape) / n_iter

        self.optimizer['arch'].update()
        monitor.update('reward', self._avg_reward, self.conf['batch_size'])

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
        for m in self.arch_modules:
            m._update_active_idx()
        self.model.apply(training=True)
        ph = self.placeholder['model']
        image = ut.image_augmentation(ph['input'])
        ph['output'] = self.model(image).apply(persistent=True)
        ph['loss'] = self.criteria(ph['output'], ph['target']) / self.iter
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
        for m in self.arch_modules:
            m._update_active_idx()
        self.model.apply(training=False)
        ph = self.placeholder['arch']
        ph['output'] = self.model(ph['input']).apply(persistent=True)
        ph['loss'] = self.criteria(ph['output'], ph['target']) / self.iter
        ph['err'] = self.evaluate(
            ph['output'].get_unlinked_variable(),
            ph['target']
        )
        ph['loss'].apply(persistent=True)
        ph['err'].apply(persistent=True)
        # setup parameters
        if len(self.optimizer['arch'].get_parameters()) == 0:
            self.optimizer['arch'].set_parameters(
                self.model.get_arch_parameters(grad_only=True)
            )
