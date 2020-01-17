import os

import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
import nnabla.utils.learning_rate_scheduler as LRS
from nnabla.logger import logger

import nnabla_nas.utils as ut
from nnabla_nas.dataset import DataLoader
from nnabla_nas.dataset.cifar10 import cifar10
from nnabla_nas.optimizer import Optimizer

from ..visualization import visualize


class Searcher(object):

    def __init__(self, model, conf):
        # dataset configuration
        data = cifar10(conf['batch_size_train'], True)
        # list of transformers
        train_transform, valid_transform = ut.dataset_transformer(conf)
        split = int(conf['train_portion'] * data.size)
        self.train_loader = DataLoader(
            data.slice(rng=None, slice_start=0, slice_end=split),
            train_transform
        )
        self.valid_loader = DataLoader(
            data.slice(rng=None, slice_start=split, slice_end=data.size),
            valid_transform
        )

        # solver configurations
        model_solver = S.__dict__[conf['model_optim']](conf['model_lr'])
        arch_solver = S.__dict__[conf['arch_optim']](conf['arch_lr'],
                                                     beta1=0.5, beta2=0.999)
        lr_scheduler = LRS.__dict__[conf['model_lr_scheduler']](
            conf['model_lr'],
            conf['epoch']*len(self.train_loader)//conf['batch_size']
        )
        self.model_optim = Optimizer(
            solver=model_solver,
            grad_clip=conf['model_grad_clip_value'] if
            conf['model_with_grad_clip'] else None,
            weight_decay=conf['model_weight_decay'],
            lr_scheduler=lr_scheduler
        )
        self.arch_optim = Optimizer(
            solver=arch_solver,
            grad_clip=conf['arch_grad_clip_value'] if
            conf['arch_with_grad_clip'] else None,
            weight_decay=conf['arch_weight_decay']
        )

        self.model = model
        self.criteria = lambda o, t: F.mean(F.softmax_cross_entropy(o, t))
        self.conf = conf

    def run(self):
        """Run the training process."""
        conf = self.conf
        model = self.model
        model_optim = self.model_optim
        arch_optim = self.arch_optim
        model_path = os.path.join(conf['model_save_path'], conf['model_name'])
        arch_file = model_path + '.json'
        log_path = os.path.join(conf['monitor_path'], 'search_config.json')

        train_size = conf['batch_size_train']
        valid_size = conf['batch_size_valid']
        batch_size = conf['batch_size']
        one_train_epoch = len(self.train_loader) // batch_size
        warmup = conf['warmup']
        train_micros = batch_size // train_size
        valid_micros = batch_size // valid_size

        # monitor the training process
        monitor = ut.get_standard_monitor(
            one_train_epoch, conf['monitor_path'])

        logger.info('Experimental settings are saved to ' + log_path)
        ut.write_to_json_file(content=conf, file_path=log_path)

        arch_modules = model.get_arch_modues()  # avoid run through all modules

        # build a graph for training
        model.train()
        train_input = nn.Variable(model.input_shape)
        train_target = nn.Variable((train_size, 1))
        train_out = model(ut.image_augmentation(train_input))
        train_loss = self.criteria(train_out, train_target)/train_micros
        train_out.persistent = True
        train_loss.persistent = True

        # assigning parameters
        model_optim.set_parameters(
            params=model.get_net_parameters(grad_only=True),
            reset=False, retain_state=True
        )
        arch_optim.set_parameters(
            params=model.get_arch_parameters(grad_only=True),
            reset=False, retain_state=True
        )

        # input and target variables for validating
        model.eval()
        valid_input = nn.Variable((valid_size, ) + model.input_shape[1:])
        valid_target = nn.Variable((valid_size, 1))
        valid_out = model(valid_input)
        valid_loss = self.criteria(valid_out, valid_target)/valid_micros
        valid_out.persistent = True
        valid_loss.persistent = True

        # whether we need to sample everytime
        requires_sample = conf['mode'] == 'sample'

        for cur_epoch in range(conf['epoch']):
            monitor.reset()

            for i in range(one_train_epoch):
                curr_iter = i + one_train_epoch * cur_epoch
                if requires_sample:
                    # sample an architecture
                    for m in arch_modules:
                        m._update_active_idx()

                # model update
                model_optim.zero_grad()
                for _ in range(train_micros):
                    train_input.d, train_target.d = self.train_loader.next()
                    train_loss.forward(clear_no_need_grad=True)
                    train_loss.backward(clear_buffer=True)
                    error = ut.categorical_error(train_out.d, train_target.d)
                    loss = train_loss.d.copy()
                    monitor['train_loss'].update(loss*train_micros, train_size)
                    monitor['train_err'].update(error, train_size)
                model_optim.update(curr_iter)

                # architecture update
                arch_optim.zero_grad()
                for _ in range(valid_micros):
                    valid_input.d, valid_target.d = self.valid_loader.next()
                    valid_loss.forward(clear_no_need_grad=True)
                    error = ut.categorical_error(valid_out.d, valid_target.d)
                    loss = valid_loss.d.copy()
                    monitor['valid_loss'].update(loss*valid_micros, valid_size)
                    monitor['valid_err'].update(error, valid_size)
                    if warmup == 0 and not requires_sample:
                        valid_loss.backward(clear_buffer=True)
                if warmup == 0:
                    if requires_sample:
                        # compute gradients
                        for m in arch_modules:
                            m._update_alpha_grad()
                        # perform control variate
                        for v in arch_optim.get_parameters().values():
                            v.g *= loss - conf['control_variate']
                    arch_optim.update(curr_iter)

                if i % conf['print_frequency'] == 0:
                    monitor.display(i)

            # saving the architecture parameters
            if conf['shared_params']:
                ut.save_dart_arch(model, arch_file)
                if conf['visualize']:
                    curr_arch = visualize(arch_file, conf['monitor_path'])
                    for tag, img in curr_arch.items():
                        monitor.write_image(tag, img, cur_epoch)
            else:
                model.save_parameters(
                    model_path + '.h5', model.get_arch_parameters())
            warmup -= warmup > 0

            monitor.write(cur_epoch)
            logger.info('Epoch %d: lr=%.5f\twu=%d\tErr=%.3f\tLoss=%.3f' %
                        (cur_epoch, model_optim.get_learning_rate(curr_iter),
                         warmup, monitor['valid_err'].avg,
                         monitor['valid_loss'].avg))

        monitor.close()

        return self
