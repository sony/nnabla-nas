# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import random
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
from nnabla_ext.cuda import clear_memory_cache

from ... import contrib
from .search import Searcher

from ...contrib.common.ofa.utils.random_resize_crop import OFAResize
from ...contrib.common.ofa.elastic_nn.utils import set_running_statistics


class OFASearcher(Searcher):
    r"""An implementation of OFA."""

    def __init__(self, model, optimizer, regularizer, dataloader, args):
        super().__init__(model, optimizer, regularizer, dataloader, args)

        manual_seed = 0
        nn.seed(manual_seed)
        np.random.seed(manual_seed)
        random.seed(manual_seed)

        self.bs_test = self.bs_valid
        self.mbs_test = self.mbs_valid
        self.accum_test = self.bs_test // self.mbs_test
        self.one_epoch_test = len(self.dataloader['test']) // self.bs_test

        self.image_size_list = self.args['train_image_size_list']
        OFAResize.IMAGE_SIZE_LIST = self.image_size_list
        OFAResize.ACTIVE_SIZE = max(self.image_size_list)
        OFAResize.IMAGE_SIZE_SEG = 4 if 'image_size_seg' not in self.args else self.args['image_size_seg']
        OFAResize.CONTINUOUS = True if "image_size_continuous" not in self.args else self.args['image_size_continuous']

        if self.args['lambda_kd'] > 0:  # knowledge distillation
            name, attributes = list(self.args['teacher_network'].items())[0]
            self.teacher_model = contrib.__dict__[name].TrainNet(**attributes)

        self.update_graph('valid')
        self.metrics = {
            k: nn.NdArray.from_numpy_array(np.zeros((1,)))
            for k in self.placeholder['valid']['metrics']
        }
        # loss and metric
        self.loss = nn.NdArray.from_numpy_array(np.zeros((1,)))

    def run(self):
        r"""Run the training process."""
        self.callback_on_start()

        # Test for init parameters
        if self.args['task'] != 'fullnet':
            OFAResize.IS_TRAINING = False
            for genotype in self.args['valid_genotypes']:
                for img_size in self.args['valid_image_size_list']:
                    self.monitor.reset()
                    OFAResize.ACTIVE_SIZE = img_size
                    self.model.set_valid_arch(genotype)
                    self.reset_running_statistics()
                    for i in tqdm(range(self.one_epoch_test), desc='Test for init parameters'):
                        self.update_graph('test')
                        self.valid_on_batch(is_test=True)
                        clear_memory_cache()
                    self.monitor.info(f'img_size={img_size}, genotype={genotype} \n')
                    self.callback_on_epoch_end(is_test=True)

                    self.loss.zero()
                    for k in self.metrics:
                        self.metrics[k].zero()

        # training
        for self.cur_epoch in range(self.cur_epoch, self.args['epoch']):
            self.monitor.reset()
            OFAResize.IS_TRAINING = True

            lr = self.optimizer['train'].get_learning_rate()
            self.monitor.info(f'Running epoch={self.cur_epoch}\tlr={lr:.5f}\n')

            OFAResize.EPOCH = self.cur_epoch
            for i in range(self.one_epoch_train):
                self.train_on_batch(self.cur_epoch, i)
                if i % (self.args['print_frequency']) == 0:
                    train_keys = [m.name for m in self.monitor.meters.values()
                                  if 'train' in m.name]
                    self.monitor.display(i, key=train_keys)
                clear_memory_cache()
            if self.cur_epoch % self.args["validation_frequency"] == 0:
                OFAResize.IS_TRAINING = False
                for genotype in self.args['valid_genotypes']:
                    for img_size in self.args['valid_image_size_list']:
                        self.monitor.reset()
                        OFAResize.ACTIVE_SIZE = img_size
                        self.model.set_valid_arch(genotype)
                        self.reset_running_statistics()
                        for i in tqdm(range(self.one_epoch_valid),
                                      desc=f'Valid [{self.cur_epoch}/{self.args["epoch"]}]'):
                            self.update_graph('valid')
                            self.valid_on_batch(is_test=False)
                            clear_memory_cache()
                        self.monitor.info(f'img_size={img_size}, genotype={genotype} \n')
                        self.callback_on_epoch_end(is_test=False)
                        self.monitor.write(self.cur_epoch)

                    self.loss.zero()
                    for k in self.metrics:
                        self.metrics[k].zero()
        return self

    def callback_on_start(self):
        keys = self.args['no_decay_keys']
        net_params = [
            self.get_net_parameters_with_keys(keys, mode='exclude', grad_only=True),  # parameters with weight decay
            self.get_net_parameters_with_keys(keys, mode='include', grad_only=True),  # parameters without weight decay
        ]
        self.optimizer['train'].set_parameters(net_params[0])
        self.optimizer['train_no_decay'].set_parameters(net_params[1])

        # load checkpoint if available
        self.load_checkpoint()

        if self.comm.n_procs > 1:
            self._grads_net = [x.grad for x in net_params[0].values()]
            self._grads_no_decay_net = [x.grad for x in net_params[1].values()]
            self.event.default_stream_synchronize()

    def train_on_batch(self, epoch, n_iter, key='train'):
        r"""Update the model parameters."""
        OFAResize.BATCH = n_iter
        batch = [self.dataloader['train'].next()
                 for _ in range(self.accum_train)]
        bz, p = self.mbs_train, self.placeholder['train']
        if key == 'train':
            self.optimizer['train'].zero_grad()
            self.optimizer['train_no_decay'].zero_grad()
        else:
            self.optimizer[key].zero_grad()

        if self.comm.n_procs > 1:
            self.event.default_stream_synchronize()

        self.update_graph(key)
        for _, data in enumerate(batch):
            self._load_data(p, data)
            p['loss'].forward(clear_no_need_grad=True)
            for k, m in p['metrics'].items():
                m.forward(clear_buffer=True)
                self.monitor.update(f'{k}/train', m.d.copy(), bz)
            p['loss'].backward(clear_buffer=True)
            loss = p['loss'].d.copy()
            self.monitor.update('loss/train', loss * self.accum_train, bz)

        if self.comm.n_procs > 1:
            self.comm.all_reduce(self._grads_net, division=True, inplace=False)
            self.comm.all_reduce(self._grads_no_decay_net, division=True, inplace=False)
        self.event.add_default_stream_event()

        if key != 'train':
            self.optimizer[key].update()
        else:
            self.optimizer['train'].update()
            self.optimizer['train_no_decay'].update()

    def valid_on_batch(self, is_test=False):
        r"""Updates the architecture parameters."""
        key = 'test' if is_test else 'valid'

        bz = self.mbs_test if is_test else self.mbs_valid
        accum = self.accum_test if is_test else self.accum_valid
        p = self.placeholder['valid']

        if self.comm.n_procs > 1:
            self.event.default_stream_synchronize()
        for _ in range(accum):
            self._load_data(p, self.dataloader[key].next())
            p['loss'].forward(clear_buffer=True)
            for k, m in p['metrics'].items():
                m.forward(clear_buffer=True)
                self.metrics[k].data += m.d.copy() * bz
            loss = p['loss'].d.copy()
            self.loss.data += loss * accum * bz

        if self.comm.n_procs > 1:
            self.comm.all_reduce(
                [self.loss] + list(self.metrics.values()), division=True, inplace=False)
        self.event.add_default_stream_event()

    def callback_on_epoch_end(self, epoch=None, is_test=False, info=None):
        if is_test:
            num_of_samples = self.one_epoch_test * self.accum_test * self.mbs_test
        else:
            num_of_samples = self.one_epoch_valid * self.accum_valid * self.mbs_valid

        self.loss.data /= num_of_samples

        for k in self.metrics:
            self.metrics[k].data /= num_of_samples
        if self.comm.rank == 0:
            self.monitor.update('loss/valid', self.loss.data[0], 1)
            self.monitor.info(f'loss/valid={self.loss.data[0]:.4f}\n')
            for k in self.metrics:
                self.monitor.update(f'{k}/valid', self.metrics[k].data[0], 1)
                self.monitor.info(f'{k}/valid={self.metrics[k].data[0]:.4f}\n')
            if info:
                self.monitor.info(f'{info}\n')
            if self.args['save_nnp']:
                self.model.save_net_nnp(
                    self.args['output_path'],
                    self.placeholder['valid']['inputs'][0],
                    self.placeholder['valid']['outputs'][0],
                    save_params=self.args.get('save_params'))
            else:
                self.model.save_parameters(
                    path=os.path.join(self.args['output_path'], 'weights.h5')
                )
            if not is_test:
                self.save_checkpoint()

    def update_graph(self, key='train'):
        r"""Builds the graph and update the placeholder.

        Args:
            key (str, optional): Type of graph. Defaults to 'train'.
        """
        assert key in ('train', 'valid', 'test')
        self.model.apply(training=key not in ['valid', 'test'])
        if self.args['lambda_kd'] > 0:
            self.teacher_model.apply(training='train')

        fake_key = 'train' if key == 'train' else 'valid'
        p = self.placeholder[fake_key]
        resize = OFAResize()
        transform = self.dataloader[fake_key].transform(fake_key)
        accum = self.accum_test if key == 'test' else (self.accum_valid if key == 'valid' else self.accum_train)

        # outputs
        inputs = [resize(transform(x)) for x in p['inputs']]
        outputs = self.model(*inputs)
        outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
        p['outputs'] = [x.apply(persistent=True) for x in outputs]

        if fake_key == 'valid' and self.args['valid_ce_loss']:
            # cross entropy loss
            p['loss'] = F.mean(F.softmax_cross_entropy(p['outputs'][0], p['targets'][0])) / accum
        else:
            if self.args['lambda_kd'] > 0:
                with nn.no_grad():
                    soft_logits = self.teacher_model(*inputs)
                    soft_logits = soft_logits if isinstance(soft_logits, (tuple, list)) else (soft_logits,)
                    p['soft_logits'] = [x.apply(need_grad=False) for x in soft_logits]
                kd_loss = self.model.kd_loss(
                    p['outputs'], p['soft_logits'], p['targets'], self.args['loss_weights'])

            # loss function
            if self.args['lambda_kd'] > 0:
                p['loss'] = (self.model.loss(p['outputs'], p['targets'], self.args['loss_weights'])
                             + self.args['lambda_kd'] * kd_loss) / accum
            else:
                p['loss'] = self.model.loss(p['outputs'], p['targets'], self.args['loss_weights']) / accum
        p['loss'].apply(persistent=True)

        # metrics to monitor during training
        targets = [out.get_unlinked_variable().apply(need_grad=False) for out in p['outputs']]
        p['metrics'] = self.model.metrics(targets, p['targets'])
        for v in p['metrics'].values():
            v.apply(persistent=True)

    def get_net_parameters_with_keys(self, keys, mode='include', grad_only=False):
        r"""Returns an `OrderedDict` containing model parameters.

        Args:
            keys (list of str): Patterns of parameters to be considered for inclusion
                or exclusion. Note: Keys passed must be in regular expression format.
            mode (str, optional): Mode of getting network parameters with keys.
                - Selects parameters satisfying the keys if mode=='include'
                - Selects parameters not satisfying the keys if mode=='exclude'
                Choices: ['include', 'exclude']. Defaults to 'include'.
            grad_only (bool, optional): If sets to `True`, then only parameters
                with `need_grad=True` are returned. Defaults to False.

        Returns:
            OrderedDict: A dictionary containing parameters.
        """

        pattern = re.compile('|'.join(keys))  # compile the pattern of all keys
        net_params = self.model.get_net_parameters(grad_only)
        if mode == 'include':  # without weight decay
            param_dict = OrderedDict()
            for name in net_params.keys():
                if re.search(pattern, name) is not None:
                    param_dict[name] = net_params[name]
            return param_dict
        elif mode == 'exclude':  # with weight decay
            param_dict = OrderedDict()
            for name in net_params.keys():
                if re.search(pattern, name) is None:
                    param_dict[name] = net_params[name]
            return param_dict
        else:
            raise ValueError('do not support %s' % mode)

    def reset_running_statistics(self, net=None, subset_size=2000, subset_batch_size=200,
                                 dataloader=None, dataloader_batch_size=None, inp_shape=None):
        if net is None:
            net = self.model
        if dataloader is None:
            subset_train_dataloader = self.dataloader['train']
            dataloader_batch_size = self.mbs_train
        if inp_shape is None:
            inp_shape = self.args['input_shapes']
        set_running_statistics(net, subset_train_dataloader, dataloader_batch_size,
                               subset_size, subset_batch_size, inp_shape)

    def callback_on_finish(self):
        pass
