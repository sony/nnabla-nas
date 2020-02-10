import os
from collections import OrderedDict

import nnabla as nn
import nnabla.utils.learning_rate_scheduler as LRS
from nnabla.logger import logger

from nnabla_nas import dataset
from nnabla_nas import utils as ut
from nnabla_nas.contrib.pnas import estimator as EST
from nnabla_nas.dataset import DataLoader
from nnabla_nas.optimizer import Optimizer


class Configuration(object):

    def __init__(self, conf=None):
        conf = conf or dict()

        # data set
        conf['dataset'] = conf.get('dataset', 'cifar10')
        self.dataset = conf['dataset']

        # training epochs
        conf['epoch'] = conf.get('epoch', 150)
        self.epoch = conf['epoch']

        # batchsize training
        conf['batch_size_train'] = conf.get('batch_size_train', 256)
        self.bs_train = conf['batch_size_train']

        # batchsize valid
        conf['batch_size_valid'] = conf.get('batch_size_valid', 256)
        self.bs_valid = conf['batch_size_valid']

        # mini batchsize training
        conf['mini_batch_train'] = conf.get('mini_batch_train', 64)
        self.mbs_train = conf['mini_batch_train']

        # mini batchsize valid
        conf['mini_batch_valid'] = conf.get('mini_batch_valid', 64)
        self.mbs_valid = conf['mini_batch_valid']

        # input shape
        conf['input_shape'] = conf.get('input_shape', (3, 32, 32))
        self.input_shape = conf['input_shape']

        assert self.bs_train % self.mbs_train == 0
        assert self.bs_valid % self.mbs_valid == 0

        # number of epochs for warming up
        conf['warmup'] = conf.get('warmup', 0)
        self.warmup = conf['warmup']

        # frequency of messages
        conf['print_frequency'] = conf.get('print_frequency', 20)
        self.print_frequency = conf['print_frequency']

        # training portion
        conf['train_portion'] = conf.get('train_portion', 1)
        self.train_portion = conf['train_portion']

        # output path
        conf['output_path'] = conf.get('output_path', 'log')
        self.output_path = conf['output_path']

        # cutout
        conf['cutout'] = conf.get('cutout', 0)
        self.cutout = conf['cutout']

        # auxiliar weights
        conf['aux_weight'] = conf.get('aux_weight', 0)
        self.aux_weight = conf['aux_weight']

        self.conf = conf

    def parse(self):
        r"""Returns a dict containing options for a Runner."""
        conf = self.conf.copy()
        options = dict()

        # define contraints
        parser = RegularizerParser(self)
        options['regularizer'] = parser.parse(conf.get('regularizer', dict()))
        conf['regularizer'] = parser.summary().copy()

        # define dataloader for training and validating
        parser = DataloaderParser(self)
        options['dataloader'] = parser.parse(conf)

        # define optimizer
        parser = OptimizerParser(self, len(options['dataloader']['train']))
        options['optimizer'] = parser.parse(conf.get('optimizer', dict()))
        conf['optimizer'] = parser.summary().copy()

        # a placeholder to store input and output variables
        parser = PlaceholderParser(self)
        options['placeholder'] = parser.parse(conf.get('placeholder', dict()))

        if not os.path.isdir(conf['output_path']):
            os.makedirs(conf['output_path'])
        file = os.path.join(conf['output_path'], 'config.json')
        logger.info(f'Saving the configurations to {file}')
        ut.write_to_json_file(conf, file)

        return options


class OptionParser(object):
    def __init__(self, options):
        self.options = options
        self.conf = dict()

    def parse(self, args: dict):
        raise NotImplementedError

    def summary(self):
        return self.conf


class OptimizerParser(OptionParser):
    def __init__(self, configuration, n):
        self.conf = configuration
        self.iter_train = n * self.conf.epoch // self.conf.bs_train
        self.iter_warm = n * self.conf.warmup // self.conf.bs_train

    def parse(self, args):
        conf = args.copy()

        optimizer = dict()
        for key in ['train', 'valid', 'warmup']:
            conf[key] = conf.get(key, dict())
            optim = conf[key]
            optim['solver'] = optim.get('solver', dict())
            optim['lr_scheduler'] = optim.get('lr_scheduler', None)

            lr_scheduler = None

            if optim['lr_scheduler'] is not None:
                scheduler_params = optim.pop('lr_scheduler').copy()
                class_name = scheduler_params.pop('name')
                lr_scheduler = LRS.__dict__[class_name](**scheduler_params)
            elif key != 'valid':
                optim['lr_scheduler'] = dict()
                optim['lr_scheduler']['name'] = 'CosineScheduler'
                optim['lr_scheduler']['lr'] = optim['solver'].get('lr', 0.01)
                optim['lr_scheduler']['max_iter'] = (
                    self.iter_train if key == 'train' else self.iter_warm
                )
                lr_scheduler = LRS.__dict__['CosineScheduler'](
                    init_lr=optim['lr_scheduler']['lr'],
                    max_iter=optim['lr_scheduler']['max_iter']
                )
            args = optim.copy()
            optim['solver']['name'] = optim['solver'].get('name', 'Sgd')
            sol = optim['solver'].copy()

            optimizer[key] = Optimizer(
                retain_state=True,
                weight_decay=args.pop('weight_decay', None),
                grad_clip=args.pop('grad_clip', None),
                lr_scheduler=lr_scheduler,
                name=sol.pop('name', 'Sgd'), **sol
            )
        self.conf = conf
        return optimizer


class RegularizerParser(OptionParser):

    def parse(self, args):
        conf = args.copy()
        regularizer = dict()
        for k, v in args.items():
            opt = v.copy()
            regularizer[k] = dict()
            regularizer[k]['bound'] = opt.pop('bound', 0)
            regularizer[k]['weight'] = opt.pop('weight', 0)
            v['bound'] = regularizer[k]['bound']
            v['weight'] = regularizer[k]['weight']
            v['name'] = opt.pop('name', 'LatencyEstimator')
            regularizer[k]['reg'] = EST.__dict__[v['name']](**opt)
        self.conf = conf
        return regularizer


class PlaceholderParser(OptionParser):

    def parse(self, conf):
        opts = self.options
        placeholder = OrderedDict({
            'train': {
                'input': nn.Variable((opts.mbs_train, *opts.input_shape)),
                'target': nn.Variable((opts.mbs_train, 1))
            },
            'valid': {
                'input': nn.Variable((opts.mbs_valid, *opts.input_shape)),
                'target': nn.Variable((opts.mbs_valid, 1))
            }
        })
        return placeholder


class DataloaderParser(OptionParser):

    def parse(self, conf):
        opts = self.options
        # dataset configuration
        if conf['search']:
            data_train, data_valid = dataset.__dict__[conf.get('dataset')](
                batch_size=(opts.mbs_train, opts.mbs_valid),
                portion=opts.train_portion,
                shuffle=True
            )
        else:
            data_train = dataset.__dict__[opts.dataset](opts.mbs_train, True)
            data_valid = dataset.__dict__[opts.dataset](opts.mbs_valid, False)

        train_transform, valid_transform = ut.dataset_transformer(conf)

        return {
            'train': DataLoader(data_train, train_transform),
            'valid': DataLoader(data_valid, valid_transform)
        }
