import os
from collections import Counter, OrderedDict

import nnabla as nn
import nnabla.functions as F
import nnabla.utils.learning_rate_scheduler as LRS
from nnabla.logger import logger

from .. import utils as ut
from ..contrib.darts.modules import CANDIDATE_FUNC
from ..contrib.pnas import estimator as EST
from ..dataset import DataLoader
from ..dataset.cifar10 import cifar10
from ..optimizer import Optimizer
from ..visualization import visualize


class Searcher(object):
    """
    Searching the best architecture.
    """

    def __init__(self, model, conf):
        self.model = model
        self.arch_modules = model.get_arch_modules()
        self.conf = conf
        self.criteria = lambda o, t: F.mean(F.softmax_cross_entropy(o, t))
        self.evaluate = lambda o, t:  F.mean(F.top_n_error(o, t))
        self.w_micros = self.conf['batch_size'] // self.conf['mini_batch_size']
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
                retain_state=conf['network']['name'] == 'pnas',
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
        model = self.model
        optim = self.optimizer
        one_epoch = len(self.loader['model']) // conf['batch_size']

        out_path = conf['output_path']
        model_path = os.path.join(out_path, conf['model_name'])
        log_path = os.path.join(out_path, 'search_config.json')
        arch_file = model_path + '.json'
        warmup = conf['warmup']
        need_resample = conf['network']['name'] == 'pnas'

        # monitor the training process
        monitor = ut.ProgressMeter(one_epoch, path=out_path)
        logger.info('Experimental settings are saved to ' + log_path)
        ut.write_to_json_file(content=conf, file_path=log_path)

        # sample computational graphs
        self._sample(verbose=True)

        for cur_epoch in range(conf['epoch']):
            monitor.reset()

            for i in range(one_epoch):
                if need_resample:
                    self._sample()

                reward = 0
                for mode, ph in self.placeholder.items():
                    optim[mode].zero_grad()
                    training = (mode == 'model')

                    for _ in range(self.w_micros):
                        ph['input'].d, ph['target'].d = self.loader[mode].next()
                        ph['loss'].forward(clear_no_need_grad=True)
                        ph['err'].forward(clear_buffer=True)
                        if training or not need_resample:
                            ph['loss'].backward(clear_buffer=True)

                        error = ph['err'].d.copy()
                        loss = ph['loss'].d.copy()

                        monitor.update(mode + '_loss', loss * self.w_micros,
                                       conf['mini_batch_size'])
                        monitor.update(mode + '_err', error,
                                       conf['mini_batch_size'])

                        # compute reward for reinfoce update
                        reward += (1-training) * loss

                    # update the model parameters or arch parameters for DARTS
                    if training or (warmup == 0 and not need_resample):
                        optim[mode].update()

                if need_resample:
                    if self.reg:
                        for k, v in self.reg.items():
                            value = v['reg'].get_estimation(model)
                            reward += v['weight']*max(0, value - v['bound'])
                            monitor.update(k, value, 1)
                    if warmup == 0:
                        self._reinforce_update(reward)
                        optim['arch'].update()

                if i % conf['print_frequency'] == 0:
                    monitor.display(i)

            warmup -= warmup > 0
            # saving the architecture parameters
            if conf['network']['name'] != 'pnas':
                ut.save_dart_arch(model, arch_file)
                for tag, img in visualize(arch_file, out_path).items():
                    monitor.write_image(tag, img, cur_epoch)
            else:
                nn.save_parameters(model_path + '.h5',
                                   model.get_arch_parameters())
                logger.info(self._get_statistics())

            monitor.write(cur_epoch)
            logger.info('Epoch %d: lr=%.5f\tErr=%.3f\tLoss=%.3f' %
                        (cur_epoch, optim['model'].get_learning_rate(),
                         monitor['arch_err'].avg, monitor['arch_loss'].avg))

        monitor.close()

        return self

    def _get_statistics(self):
        stats = ''
        ans = Counter([m._active for m in self.arch_modules])
        total = len(self.arch_modules)
        for k in range(len(self.op_names)):
            name = self.op_names[k]
            stats += name + f' = {ans[k]/total*100:.2f}%\t'
        return stats

    def _reinforce_update(self, reward):
        for m in self.arch_modules:
            m._update_alpha_grad()
        # perform control variate
        for v in self.optimizer['arch'].get_parameters().values():
            v.g *= reward - self.conf['arch_optimizer']['control_variate']

    def _sample(self, verbose=False):
        """Sample new graphs, one for model training and one for arch training."""
        if self.conf['network']['name'] == 'pnas':
            for m in self.arch_modules:
                m._update_active_idx()

        for mode, ph in self.placeholder.items():
            training = (mode == 'model')
            self.model.apply(training=training)

            # loss and error
            image = ut.image_augmentation(ph['input'])
            ph['output'] = self.model(image).apply(persistent=True)
            ph['loss'] = self.criteria(
                ph['output'], ph['target']) / self.w_micros
            ph['err'] = self.evaluate(
                ph['output'].get_unlinked_variable(), ph['target'])
            ph['loss'].apply(persistent=True)
            ph['err'].apply(persistent=True)

            # set parameters to the optimizer
            params = self.model.get_net_parameters(grad_only=True) if training\
                else self.model.get_arch_parameters(grad_only=True)
            self.optimizer[mode].set_parameters(params)

        if verbose:
            model_size = ut.get_params_size(
                self.optimizer['model'].get_parameters())/1e6
            arch_size = ut.get_params_size(
                self.optimizer['arch'].get_parameters())/1e6
            logger.info('Model size={:.6f} MB\t Arch size={:.6f} MB'.format(
                model_size, arch_size))

        return self
