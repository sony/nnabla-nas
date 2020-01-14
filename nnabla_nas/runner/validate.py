import os

import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
import nnabla.utils.learning_rate_scheduler as LRS
from nnabla.ext_utils import get_extension_context
from nnabla.logger import logger
from tqdm import tqdm

import nnabla_nas.utils as ut
from nnabla_nas.dataset import DataLoader
from nnabla_nas.dataset.cifar10 import cifar10
from nnabla_nas.optimizer import Optimizer


class Trainer(object):

    def __init__(self, model, conf):
        # dataset loader setup
        train_transform, valid_transform = ut.dataset_transformer(conf)
        self.train_loader = DataLoader(
            cifar10(conf['batch_size_train'], True, shuffle=True),
            train_transform
        )
        self.valid_loader = DataLoader(
            cifar10(conf['batch_size_valid'], False, shuffle=False),
            valid_transform
        )

        # solver configurations
        model_solver = S.__dict__[conf['model_optim']](conf['model_lr'])
        max_iter = conf['epoch'] * len(self.train_loader) // conf['batch_size']
        lr_scheduler = LRS.__dict__[conf['model_lr_scheduler']](
            conf['model_lr'],
            max_iter=max_iter  # this is for CosineScheduler
        )
        self.optimizer = Optimizer(
            solver=model_solver,
            grad_clip=conf['model_grad_clip_value'] if
            conf['model_with_grad_clip'] else None,
            weight_decay=conf['model_weight_decay'],
            lr_scheduler=lr_scheduler
        )

        if not conf['shared_params']:
            model.load_parameters(conf['arch'] + '.h5')
            # preparing to sample one graph
            for module in model.get_arch_modues():
                module._mode = 'max'
                module._update_active_idx()

        self.model = model
        self.criteria = lambda o, t: F.mean(F.softmax_cross_entropy(o, t))
        self.conf = conf

    def run(self):
        """Run the training process."""
        conf = self.conf
        model = self.model
        optimizer = self.optimizer
        criteria = self.criteria
        drop_prob = self.model._drop_prob
        save_path = os.path.join(conf['model_save_path'], conf['model_name'])

        valid_size = conf['batch_size_valid']
        train_size = conf['batch_size_train']
        batch_size = conf['batch_size']

        assert len(self.valid_loader) % valid_size == 0

        n_micros = batch_size // train_size
        one_train_epoch = len(self.train_loader) // batch_size
        one_valid_epoch = len(self.valid_loader) // valid_size
        aux_weight = conf['auxiliary_weight'] / n_micros

        # monitor the training process
        monitor = ut.get_standard_monitor(
            one_train_epoch, conf['monitor_path'])

        # write out the configuration
        log_path = os.path.join(conf['monitor_path'], 'train_config.json')
        logger.info('Experimental settings are saved to ' + log_path)
        ut.write_to_json_file(content=conf, file_path=log_path)
        # setup context for nnabla
        ctx = get_extension_context(
            conf['context'], device_id=conf['device_id'])
        nn.set_default_context(ctx)

        # sample one graph for training
        model.train()
        train_input = nn.Variable(model.input_shape)
        train_target = nn.Variable((train_size, 1))
        train_out, aux_out = model(ut.image_augmentation(train_input))
        train_loss = criteria(train_out, train_target)/n_micros
        if conf['auxiliary']:
            train_loss += aux_weight*criteria(aux_out, train_target)
        train_loss.persistent = True
        train_out.persistent = True

        # assign parameters
        optimizer.set_parameters(
            params=model.get_net_parameters(grad_only=True),
            reset=False, retain_state=True
        )

        # print a summary
        model_size = ut.get_params_size(optimizer.get_parameters())
        aux_size = ut.get_params_size(model._auxiliary_head.get_parameters())
        model_size = (model_size - aux_size) / 1e6
        logger.info('Model size = {:.6f} MB'.format(model_size))

        # sample a graph for validating
        model.eval()
        valid_input = nn.Variable((valid_size,) + model.input_shape[1:])
        valid_target = nn.Variable((valid_size, 1))
        valid_output, _ = model(valid_input)
        valid_loss = criteria(valid_output, valid_target)
        valid_output.persistent = True
        valid_loss.persistent = True
        best_error = 1.0

        for cur_epoch in range(conf['epoch']):
            monitor.reset()
            # adjusting the drop path rate
            if drop_prob:
                drop_rate = conf['drop_path_prob']*cur_epoch/conf['epoch']
                drop_prob.d[0] = drop_rate

            for i in range(one_train_epoch):
                curr_iter = i + one_train_epoch*cur_epoch
                # training model parameters
                optimizer.zero_grad()
                error = loss = 0

                # mini batches update
                for _ in range(n_micros):
                    train_input.d, train_target.d = self.train_loader.next()
                    train_loss.forward(clear_no_need_grad=True)
                    train_loss.backward(clear_buffer=True)
                    error += ut.categorical_error(train_out.d, train_target.d)
                    loss += train_loss.d.copy()

                optimizer.update(curr_iter)

                monitor['train_loss'].update(loss, train_size)
                monitor['train_err'].update(error/n_micros, train_size)
                if i % conf['print_frequency'] == 0:
                    monitor.display(i, ['train_loss', 'train_err'])

            # compute the validation error
            for i in tqdm(range(one_valid_epoch)):
                valid_input.d, valid_target.d = self.valid_loader.next()
                valid_loss.forward(clear_buffer=True)
                error = ut.categorical_error(valid_output.d, valid_target.d)
                monitor['valid_loss'].update(valid_loss.d.copy(), valid_size)
                monitor['valid_err'].update(error, valid_size)

            if monitor['valid_err'].avg < best_error:
                best_error = monitor['valid_err'].avg
                model.save_parameters(save_path)

            monitor.write(cur_epoch)
            logger.info('Epoch %d: lr=%.5f\tdp=%.5f\tErr=%.3f\tLoss=%.3f' %
                        (cur_epoch, optimizer.get_learning_rate(curr_iter),
                         drop_rate, monitor['valid_err'].avg,
                         monitor['valid_loss'].avg))

        monitor.close()

        return self
