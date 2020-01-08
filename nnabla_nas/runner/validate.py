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
        max_iter = conf['epoch'] * len(self.train_loader) // conf['batch_size']
        lr_scheduler = LRS.__dict__[conf['model_lr_scheduler']](
            conf['model_lr'],
            gamma=0.97,
            iter_steps=[max_iter*i for i in range(1, conf['epoch'] + 1)]
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
        self.conf = conf

    def run(self):
        """Run the training process."""
        conf = self.conf
        model = self.model
        model_optim = self.model_optim

        one_epoch = len(self.train_loader) // conf['batch_size']
        one_valid_epoch = len(self.valid_loader) // conf['minibatch_size']
        # monitor the training process
        monitor = ut.ProgressMeter(
            num_batches=one_epoch,
            meters=[
                ut.AverageMeter('train_loss', fmt=':5.3f'),
                ut.AverageMeter('valid_loss', fmt=':5.3f'),
                ut.AverageMeter('train_err', fmt=':5.3f'),
                ut.AverageMeter('valid_err', fmt=':5.3f')
            ],
            tb_writer=SummaryWriter(
                os.path.join(conf['monitor_path'], 'tensorboard')
            )
        )
        n_micros = conf['batch_size'] // conf['minibatch_size']

        # write out the configuration
        path = os.path.join(conf['monitor_path'], 'train_config.json')
        with open(path, 'w+') as file:
            json.dump(conf, file,  ensure_ascii=False,
                      indent=4, default=lambda o: '<not serializable>')

        ctx = get_extension_context(
            conf['context'], device_id=conf['device_id'])
        nn.set_default_context(ctx)

        # input and target variables
        x_var = nn.Variable(model.input_shape)
        image = F.random_crop(F.pad(x_var, (4, 4, 4, 4)), shape=(x_var.shape))
        image = F.image_augmentation(image, flip_lr=True)
        t_var = nn.Variable((conf['minibatch_size'], 1))

        model.train()
        for m in model.get_arch_modues():
            m._mode = 'max'
            m._update_active_idx()

        def criteria(o, t):
            return F.mean(F.softmax_cross_entropy(o, t))/n_micros

        # sample one graph for training
        out, aux = model(image)
        loss = criteria(out, t_var)
        if conf['auxiliary']:
            loss += conf['auxiliary_weight'] * criteria(aux, t_var)

        params = model.get_net_parameters(grad_only=True)
        model_optim.set_parameters(params)

        # sample a graph for validating
        model.eval()
        val_tar = nn.Variable((conf['minibatch_size'], 1))
        val_var = nn.Variable(model.input_shape)
        val_out, _ = model(val_var)
        val_loss = F.mean(F.softmax_cross_entropy(val_out, val_tar))

        for cur_epoch in range(conf['epoch']):
            monitor.reset()
            for i in range(one_epoch):
                curr_iter = i + one_epoch * cur_epoch

                # training model parameters
                model_optim.zero_grad()

                v_err = v_loss = 0
                # mini batches update
                for _ in range(n_micros):
                    x_var.d, t_var.d = self.train_loader.next()
                    loss.forward()
                    loss.backward()
                    v_err += ut.categorical_error(out.d, t_var.d)
                    v_loss += loss.d

                model_optim.update(curr_iter)
                # add info to the monitor
                monitor['train_loss'].update(v_loss)
                monitor['train_err'].update(v_err/n_micros)

                if i % conf['print_frequency'] == 0:
                    monitor.display(i, ['train_loss', 'train_err'])

            # mini batches update
            for i in tqdm(range(one_valid_epoch)):
                val_var.d, val_tar.d = self.valid_loader.next()
                val_loss.forward()
                err = ut.categorical_error(val_out.d, val_tar.d)
                # add info to the monitor
                monitor['valid_loss'].update(val_loss.d)
                monitor['valid_err'].update(err)

            # write losses and save model after each epoch
            monitor.write(cur_epoch)

            # saving the architecture parameters
            model.save_parameters(
                path=os.path.join(
                    conf['model_save_path'], conf['model_name'])
            )

        monitor.close()
        return self
