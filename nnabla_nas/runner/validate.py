import os

import nnabla as nn
import nnabla.functions as F
import nnabla.utils.learning_rate_scheduler as LRS
from nnabla.logger import logger
from tqdm import tqdm

import nnabla_nas.utils as ut
from nnabla_nas.dataset import DataLoader
from nnabla_nas.dataset.cifar10 import cifar10
from nnabla_nas.optimizer import Optimizer


class Trainer(object):

    def __init__(self, model, conf):
        self.model = model
        self.criteria = lambda o, t: F.mean(F.softmax_cross_entropy(o, t))
        self.evaluate = lambda o, t:  F.mean(F.top_n_error(o, t))
        self.conf = conf

        # dataloader
        train_transform, valid_transform = ut.dataset_transformer(conf)
        self.loader = {
            'train': DataLoader(
                cifar10(conf['batch_size_train'], True, shuffle=True),
                train_transform
            ),
            'valid': DataLoader(
                cifar10(conf['batch_size_valid'], False, shuffle=False),
                valid_transform
            )
        }

        # solver configurations
        optim = conf['optimizer'].copy()
        lr_scheduler = ut.get_object_from_dict(
            module=LRS.__dict__,
            args=optim.pop('lr_scheduler', None)
        )
        solver = optim['solver']
        self.optimizer = Optimizer(
            retain_state=False,
            weight_decay=optim.pop('weight_decay', None),
            grad_clip=optim.pop('grad_clip', None),
            lr_scheduler=lr_scheduler,
            name=solver.pop('name'), **solver
        )

    def run(self):
        """Run the training process."""
        conf = self.conf
        model = self.model
        optimizer = self.optimizer
        criteria = self.criteria
        evaluate = self.evaluate
        save_path = os.path.join(conf['output_path'], conf['model_name'])

        valid_size = conf['batch_size_valid']
        train_size = conf['batch_size_train']
        batch_size = conf['batch_size']

        assert len(self.loader['valid']) % valid_size == 0

        n_micros = batch_size // train_size
        one_train_epoch = len(self.loader['train']) // batch_size
        one_valid_epoch = len(self.loader['valid']) // valid_size

        # monitor the training process
        monitor = ut.ProgressMeter(one_train_epoch, path=conf['output_path'])

        # write out the configuration
        log_path = os.path.join(conf['output_path'], 'train_config.json')
        logger.info('Experimental settings are saved to ' + log_path)
        ut.write_to_json_file(content=conf, file_path=log_path)

        # sample one graph for training
        model.apply(training=True)
        train_input = nn.Variable((train_size, 3, 32, 32))
        train_target = nn.Variable((train_size, 1))
        if conf['auxiliary']:
            train_out, aux_out = model(ut.image_augmentation(train_input))
        else:
            train_out = model(ut.image_augmentation(train_input))
        train_out.apply(persistent=True)
        train_loss = criteria(train_out, train_target)/n_micros
        train_err = evaluate(train_out.get_unlinked_variable(), train_target)
        if model._auxiliary:
            aux_weight = conf['auxiliary_weight'] / n_micros
            train_loss += aux_weight * criteria(aux_out, train_target)
        train_loss.apply(persistent=True)
        train_err.apply(persistent=True)
        # assign parameters
        optimizer.set_parameters(model.get_parameters(grad_only=True))

        # print a summary
        model_size = ut.get_params_size(optimizer.get_parameters())
        if model._auxiliary:
            model_size -= ut.get_params_size(
                model._auxiliary_head.get_parameters())
        logger.info('Model size = {:.6f} MB'.format(model_size*1e-6))

        # sample a graph for validating
        model.apply(training=False)
        valid_input = nn.Variable((valid_size, 3, 32, 32))
        valid_target = nn.Variable((valid_size, 1))
        if conf['auxiliary']:
            valid_out, _ = model(valid_input)
        else:
            valid_out = model(valid_input)
        valid_out.apply(persistent=True)
        valid_out.apply(need_grad=False)
        valid_loss = criteria(valid_out, valid_target)
        valid_err = evaluate(valid_out.get_unlinked_variable(), valid_target)
        valid_loss.apply(persistent=True)
        valid_err.apply(persistent=True)
        best_error = 1.0

        for cur_epoch in range(conf['epoch']):
            monitor.reset()

            for i in range(one_train_epoch):
                # training model parameters
                optimizer.zero_grad()
                for _ in range(n_micros):
                    train_input.d, train_target.d = self.loader['train'].next()
                    train_loss.forward(clear_no_need_grad=True)
                    train_loss.backward(clear_buffer=True)
                    train_err.forward(clear_buffer=True)
                    loss = train_loss.d.copy()
                    monitor.update('train_loss', loss * n_micros, train_size)
                    monitor.update('train_err', train_err.d.copy(), train_size)
                optimizer.update()

                if i % conf['print_frequency'] == 0:
                    monitor.display(i, ['train_loss', 'train_err'])

            # compute the validation error
            for i in tqdm(range(one_valid_epoch)):
                valid_input.d, valid_target.d = self.loader['valid'].next()
                valid_loss.forward(clear_buffer=True)
                valid_err.forward(clear_buffer=True)
                monitor.update('valid_loss', valid_loss.d.copy(), valid_size)
                monitor.update('valid_err', valid_err.d.copy(), valid_size)

            if monitor['valid_err'].avg < best_error:
                best_error = monitor['valid_err'].avg
                nn.save_parameters(save_path, model.get_parameters())

            monitor.write(cur_epoch)
            logger.info('Epoch %d: lr=%.5f\tErr=%.3f\tLoss=%.3f' %
                        (cur_epoch, optimizer.get_learning_rate(),
                         monitor['valid_err'].avg, monitor['valid_loss'].avg))

        monitor.close()

        return self
