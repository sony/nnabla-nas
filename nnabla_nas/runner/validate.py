import json
import os

import nnabla as nn
import nnabla.functions as F
import nnabla.utils.learning_rate_scheduler as LRS
import numpy as np
from nnabla.ext_utils import get_extension_context
from nnabla.logger import logger
from tensorboardX import SummaryWriter
from tqdm import tqdm

import nnabla_nas.utils as ut
from nnabla_nas.dataset import DataLoader
from nnabla_nas.dataset.cifar10.cifar10_data import data_iterator_cifar10
from nnabla_nas.optimizer import Optimizer, Solver


class Trainer(object):

    def __init__(self, model, conf):
        # list of transformers
        train_transform, valid_transform = ut.dataset_transformer()
        # dataset configuration
        self.train_loader = DataLoader(
            data_iterator_cifar10(conf['minibatch_size'], True),
            train_transform
        )
        self.valid_loader = DataLoader(
            data_iterator_cifar10(conf['minibatch_size'], False),
            valid_transform
        )

        # solver configurations
        model_solver = Solver(conf['model_optim'], conf['model_lr'])
        one_train_epoch = len(self.train_loader) // conf['batch_size']
        lr_scheduler = LRS.__dict__[conf['model_lr_scheduler']](
            conf['model_lr'],
            gamma=0.97,
            iter_steps=[one_train_epoch*i for i in range(1, conf['epoch'] + 1)]
        )  # this is for StepScheduler

        self.model_optim = Optimizer(
            solver=model_solver,
            grad_clip=conf['model_grad_clip_value'] if
            conf['model_with_grad_clip'] else None,
            weight_decay=conf['model_weight_decay'],
            lr_scheduler=lr_scheduler
        )

        if conf['shared_params']:
            arch_params = json.load(open(conf['arch'] + '.json'))
            # assign alpha weights
            logger.info('Loading normal cells ...')
            for alpha, idx in zip(model._alpha_normal, arch_params['normal']):
                w = np.zeros(model._num_ops)
                w[idx] = 1
                alpha.d = w.reshape(alpha.d.shape)
            logger.info('Loading reduce cells ...')
            for alpha, idx in zip(model._alpha_reduce, arch_params['reduce']):
                w = np.zeros(model._num_ops)
                w[idx] = 1
                alpha.d = w.reshape(alpha.d.shape)
        else:
            model.load_parameters(conf['arch'] + '.h5')

        self.model = model
        self.criteria = lambda o, t: F.mean(F.softmax_cross_entropy(o, t))
        self.conf = conf

    def run(self):
        """Run the training process."""
        conf = self.conf
        model = self.model
        model_optim = self.model_optim
        criteria = self.criteria
        drop_prob = self.model._drop_prob

        n_micros = conf['batch_size'] // conf['minibatch_size']
        one_train_epoch = len(self.train_loader) // conf['batch_size']
        one_valid_epoch = len(self.valid_loader) // conf['minibatch_size']
        # monitor the training process
        monitor = ut.get_standard_monitor(
            one_train_epoch, conf['monitor_path'])

        # write out the configuration
        ut.write_to_json_file(
            content=conf,
            file_path=os.path.join(conf['monitor_path'], 'train_config.json')
        )

        ctx = get_extension_context(
            conf['context'], device_id=conf['device_id'])
        nn.set_default_context(ctx)

        # input and target variables
        train_input = nn.Variable(model.input_shape)
        train_target = nn.Variable((conf['minibatch_size'], 1))

        model.train()
        for module in model.get_arch_modues():
            module._mode = 'max'
            module._update_active_idx()

        # sample one graph for training
        train_out, auxilary_out = model(ut.image_augmentation(train_input))
        train_loss = criteria(train_out, train_target) / n_micros
        if conf['auxiliary']:
            train_loss += conf['auxiliary_weight'] * \
                criteria(auxilary_out, train_target) / n_micros

        train_loss.persistent = True
        train_out.persistent = True

        params = model.get_net_parameters(grad_only=True)
        model_optim.set_parameters(params)

        # sample a graph for validating
        model.eval()
        valid_input = nn.Variable(model.input_shape)
        valid_target = nn.Variable((conf['minibatch_size'], 1))
        valid_output, _ = model(valid_input)
        valid_loss = criteria(valid_output, valid_target)

        valid_output.persistent = True
        valid_loss.persistent = True
        best_error = 1.0

        for cur_epoch in range(conf['epoch']):
            monitor.reset()

            # adjusting the drop path rate
            if drop_prob:
                drop_rate = conf['drop_path_prob'] * \
                    (cur_epoch / conf['epoch']) + 1e-8
                drop_prob.d = np.array([drop_rate]).reshape(drop_prob.d.shape)

            for i in range(one_train_epoch):
                curr_iter = i + one_train_epoch * cur_epoch

                # training model parameters
                model_optim.zero_grad()
                error = loss = 0
                # mini batches update
                for _ in range(n_micros):
                    train_input.d, train_target.d = self.train_loader.next()
                    train_loss.forward(clear_no_need_grad=True)
                    train_loss.backward(clear_buffer=True)
                    error += ut.categorical_error(train_out.d, train_target.d)
                    loss += train_loss.d

                model_optim.update(curr_iter)
                # add info to the monitor
                monitor['train_loss'].update(loss)
                monitor['train_err'].update(error/n_micros)

                if i % conf['print_frequency'] == 0:
                    monitor.display(i, ['train_loss', 'train_err'])

            # mini batches update
            for i in tqdm(range(one_valid_epoch)):
                valid_input.d, valid_target.d = self.valid_loader.next()
                valid_loss.forward(clear_buffer=True)
                error = ut.categorical_error(valid_output.d, valid_target.d)
                # add info to the monitor
                monitor['valid_loss'].update(valid_loss.d)
                monitor['valid_err'].update(error)

            # write losses and save model after each epoch
            monitor.write(cur_epoch)
            
            if monitor['valid_err'].avg < best_error:
                best_error = monitor['valid_err'].avg
                # saving the architecture parameters
                logger.info('Found a better model at epoch {}'.format(cur_epoch))
                model.save_parameters(
                    path=os.path.join(
                        conf['model_save_path'], conf['model_name'])
                )

            print("LR is ", model_optim._lr_scheduler.get_learning_rate(curr_iter))

        monitor.close()

        return self
