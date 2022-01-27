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

import os
import random
import numpy as np
import json
import time

import nnabla as nn
import nnabla.functions as F

from .search import Searcher
from ...utils.helper import ProgressMeter

from ...contrib.classification.ofa.networks.mobilenet_v3 import MobileNetV3Large
from ...contrib.classification.ofa.network import SearchNet

from ...contrib.classification.ofa.ofa_modules.common_tools import cross_entropy_loss_with_soft_target
from ...contrib.classification.ofa.ofa_modules.my_modules import init_models
from ...contrib.classification.ofa.ofa_modules.my_random_resize_crop import MyResize
from ...contrib.classification.ofa.ofa_modules.dynamic_op import DynamicBatchNorm2d


class OFASearcher(Searcher):
    r"""An implementation of OFA."""
    def __init__(self, model, optimizer, dataloader, args):
        super().__init__(model, optimizer, dataloader, args)

        manual_seed = 0
        nn.seed(manual_seed)
        np.random.seed(manual_seed)
        random.seed(manual_seed)

        self.bs_test = self.bs_valid
        self.mbs_test = self.mbs_valid
        self.accum_test = self.bs_valid // self.mbs_valid
        self.one_epoch_test = len(self.dataloader['test']) // self.bs_test

        init_models(self.model, model_init='he_fout')

        self.image_size_list = [128, 160, 192, 224]
        MyResize.IMAGE_SIZE_LIST = self.image_size_list
        MyResize.CONTINUOUS = True

        # for resetting BN params when validation
        self.subset_train_dataloader = None
        self.forward_model = SearchNet(
            self.model._num_classes, self.model._bn_param, dropout=0,
            ks_list=self.model._ks_list, expand_ratio_list=self.model._expand_ratio_list,
            depth_list=self.model._depth_list
        )

        self.validate_func_dict = {
            'image_size_list': {224} if isinstance(self.image_size_list, int) else sorted({160, 224}),
            'ks_list': sorted({min(self.model._ks_list), max(self.model._ks_list)}),
            'depth_list': sorted({min(self.model._depth_list), max(self.model._depth_list)}),
            'expand_ratio_list': sorted({min(self.model._expand_ratio_list), max(self.model._expand_ratio_list)}),
            'width_mult_list': [0]
        }

        if self.args['kd_ratio'] > 0:
            self.teacher_model = MobileNetV3Large(
                num_classes=self.args['num_classes'], bn_param=(0.9, 1e-5),
                dropout=0, width_mult=1.0, ks=7, expand_ratio=6, depth_param=4,
            )
            self.teacher_model.load_ofa_parameters(
                self.args['teacher_path'], raise_if_missing=True)

        self.update_graph_ofa('valid')
        self.metrics = {
            k: nn.NdArray.from_numpy_array(np.zeros((1,)))
            for k in self.placeholder['valid']['metrics']
        }
        # loss and metric
        self.loss = nn.NdArray.from_numpy_array(np.zeros((1,)))

    def run(self):
        r"""Run the training process."""
        self.callback_on_start()

        current_epoch = 0
        best_acc = 0

        # Test for init parameter
        if self.args['task'] not in ['fullnet']:
            MyResize.IS_TRAINING = False
            for setting, name in self.subnet_settings:
                self.monitor.reset()
                MyResize.ACTIVE_SIZE = setting['img_size']
                self.model.set_active_subnet(**setting)
                self.reset_running_statistics(setting)
                for i in range(self.one_epoch_test):
                    self.update_graph_ofa('test')
                    self.valid_on_batch(is_test=True)
                self.monitor.info(f'setting={setting} \n')
                self.callback_on_epoch_end(is_test=True)

                val_err = self.metrics['error'].data[0]
                self.loss.zero()
                for k in self.metrics:
                    self.metrics[k].zero()

        self.valid_func_setup()

        # training
        end_epoch = self.args['epoch']
        for cur_epoch in range(current_epoch, end_epoch):
            self.monitor.reset()
            MyResize.IS_TRAINING = True

            self.epoch = cur_epoch
            lr = self.optimizer['train'].get_learning_rate()
            self.monitor.info(f'Running epoch={cur_epoch}\tlr={lr:.5f}\n')

            MyResize.EPOCH = cur_epoch
            for i in range(self.one_epoch_train):
                self.train_on_batch(cur_epoch, i)
                if i % (self.args['print_frequency']) == 0:
                    train_keys = [m.name for m in self.monitor.meters.values()
                                  if 'train' in m.name]
                    self.monitor.display(i, key=train_keys)
            if cur_epoch % self.args["validation_frequency"] == 0:
                MyResize.IS_TRAINING = False
                val_err_list = []
                for setting, name in self.subnet_settings:
                    self.monitor.reset()
                    MyResize.ACTIVE_SIZE = setting['img_size']
                    self.model.set_active_subnet(**setting)
                    self.reset_running_statistics(setting)
                    for i in range(self.one_epoch_valid):
                        self.update_graph_ofa('valid')
                        self.valid_on_batch(is_test=False)
                    self.monitor.info(f'setting={setting} \n')
                    self.callback_on_epoch_end(is_test=False)
                    self.monitor.write(cur_epoch)

                    val_err = self.metrics['error'].data[0]
                    self.loss.zero()
                    for k in self.metrics:
                        self.metrics[k].zero()
                    val_err_list.append(val_err)
                val_acc_epoch = round(1 - np.array(val_err_list).mean(), 5)
                if val_acc_epoch > best_acc and self.comm.rank == 0:
                    best_params_path = os.path.join(
                        self.args['output_path'],
                        f"best_weights_epoch{cur_epoch}_acc{val_acc_epoch}.h5")
                    try:
                        self.model.save_parameters(path=best_params_path)
                    except Exception as e:
                        print("error: ", e)
                        pass
                    if cur_epoch > 0:
                        if os.path.exists(previous_best_path):
                            os.remove(previous_best_path)
                    best_acc = val_acc_epoch
                    previous_best_path = best_params_path

        if self.comm.rank == 0:
            try:
                params_path = os.path.join(
                    self.args['output_path'], f"weights_epoch{cur_epoch}.h5")
                self.model.save_parameters(path=params_path)
            except Exception as e:
                print('error: ', e)

        return self

    def callback_on_start(self):
        if 'pretrained_path' in self.args:
            self.model.load_parameters(self.args['pretrained_path'])

        if self.args['task'] == 'fullnet':
            pass
        elif self.args['task'] == 'kernel':
            self.validate_func_dict['ks_list'] = sorted(self.model._ks_list.copy())
        elif self.args['task'] == 'depth':
            depth_stage_list = self.model._depth_list.copy()
            depth_stage_list.sort(reverse=True)
            self.validate_func_dict['depth_list'] = sorted(self.model._depth_list)
        elif self.args['task'] == 'expand':
            expand_stage_list = self.model._expand_ratio_list.copy()
            expand_stage_list.sort(reverse=True)
            n_stages = len(expand_stage_list) - 1
            current_stage = n_stages - 1
            self.validate_func_dict['expand_ratio_list'] = sorted(self.model._expand_ratio_list)
            # reorganize middle layers
            self.model.re_organize_middle_weights(expand_ratio_stage=current_stage)
        elif self.args['task'] == 'width_mult':
            width_stage_list = self.model._width_mult_list.copy()
            width_stage_list.sort(reverse=True)
            n_stages = len(width_stage_list) - 1
            current_stage = n_stages - 1
            # reorganize weights
            if current_stage == 0:
                self.model.re_organize_middle_weights(
                    expand_ratio_stage=len(self.model._expand_ratio_list) - 1)
        else:
            raise NotImplementedError

        # dynamic BN
        if (self.args['task'] == 'expand') or (self.args['task'] == 'width_mult'):
            DynamicBatchNorm2d.GET_STATIC_BN = False
        else:
            DynamicBatchNorm2d.GET_STATIC_BN = True

        self.subnet_settings = []
        for d in self.validate_func_dict['depth_list']:
            for e in self.validate_func_dict['expand_ratio_list']:
                for ks in self.validate_func_dict['ks_list']:
                    for w in self.validate_func_dict['width_mult_list']:
                        for img_size in self.validate_func_dict['image_size_list']:
                            self.subnet_settings.append([{
                                'img_size': img_size,
                                'd': d,
                                'e': e,
                                'ks': ks,
                                'w': w,
                            }, 'R%s-D%s-E%s-K%s-W%s' % (img_size, d, e, ks, w)])

        keys = self.args['no_decay_keys'].split('#')
        net_params = [
            self.model.get_net_parameters(keys, mode='exclude', grad_only=True),  # parameters with weight decay
            self.model.get_net_parameters(keys, mode='include', grad_only=True),  # parameters without weight decay
        ]
        self.optimizer['train'].set_parameters(net_params[0])
        self.optimizer['train_no_decay'].set_parameters(net_params[1])

        if self.comm.n_procs > 1:
            self._grads_net = [x.grad for x in net_params[0].values()]
            self._grads_no_decay_net = [x.grad for x in net_params[1].values()]
            self.event.default_stream_synchronize()

    def valid_func_setup(self):
        if self.args['task'] == 'depth':
            # add depth list constraints
            depth_stage_list = self.model._depth_list.copy()
            depth_stage_list.sort(reverse=True)
            if len(set(self.model._ks_list)) == 1 and len(set(self.model._expand_ratio_list)) == 1:
                self.validate_func_dict['depth_list'] = depth_stage_list
            else:
                self.validate_func_dict['depth_list'] = sorted({
                    min(depth_stage_list), max(depth_stage_list)})
        elif self.args['task'] == 'expand':
            expand_stage_list = self.model._expand_ratio_list.copy()
            expand_stage_list.sort(reverse=True)
            if len(set(self.model._ks_list)) == 1 and len(set(self.model._depth_list)) == 1:
                self.validate_func_dict['expand_ratio_list'] = expand_stage_list
            else:
                self.validate_func_dict['expand_ratio_list'] = sorted({
                    min(expand_stage_list), max(expand_stage_list)})
        elif self.args['task'] == 'width_mult_list':
            width_stage_list = self.model._width_mult_list.copy()
            width_stage_list.sort(reverse=True)
            self.validate_func_dict['width_mult_list'] = sorted({0, len(width_stage_list) - 1})
        else:
            pass

        self.subnet_settings = []
        for d in self.validate_func_dict['depth_list']:
            for e in self.validate_func_dict['expand_ratio_list']:
                for ks in self.validate_func_dict['ks_list']:
                    for w in self.validate_func_dict['width_mult_list']:
                        for img_size in self.validate_func_dict['image_size_list']:
                            self.subnet_settings.append([{
                                'img_size': img_size,
                                'd': d,
                                'e': e,
                                'ks': ks,
                                'w': w,
                            }, 'R%s-D%s-E%s-K%s-W%s' % (img_size, d, e, ks, w)])

    def train_on_batch(self, epoch, n_iter, key='train'):
        r"""Update the model parameters."""
        MyResize.BATCH = n_iter
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

        self.update_graph_ofa(key)
        for _, data in enumerate(batch):
            self._load_data(p, data)
            subnet_seed = int('%d%.3d%.3d' % (epoch * bz + n_iter, _, 0))
            random.seed(subnet_seed)

            subnet_settings = self.model.sample_active_subnet()
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
                self.monitor.info(f'{k}={self.metrics[k].data[0]:.4f}\n')
            if info:
                self.monitor.info(f'{info}\n')

    def update_graph_ofa(self, key='train'):
        r"""Builds the graph and update the placeholder.

        Args:
            key (str, optional): Type of graph. Defaults to 'train'.
        """
        assert key in ('train', 'valid', 'test')
        self.model.apply(training=key not in ['valid', 'test'])
        if self.args['kd_ratio'] > 0:
            self.teacher_model.apply(training='train')

        fake_key = 'train' if key == 'train' else 'valid'
        p = self.placeholder[fake_key]
        transform = self.dataloader[fake_key].transform()
        accum = self.accum_test if key == 'test' else (self.accum_valid if key == 'valid' else self.accum_train)

        # outputs
        inputs = [transform(x) for x in p['inputs']]
        outputs = self.model(*inputs)
        p['outputs'] = [outputs.apply(persistent=True)]

        if fake_key == 'valid':
            # cross entropy loss
            p['loss'] = F.mean(F.softmax_cross_entropy(p['outputs'][0], p['targets'][0])) / accum
        else:
            if self.args['kd_ratio'] > 0:
                soft_logits = self.teacher_model(*inputs)
                soft_logits.need_grad = False
                soft_label = F.softmax(soft_logits, axis=1)
                p['soft_label'] = soft_label.apply(persistent=True)
                kd_loss = cross_entropy_loss_with_soft_target(p['outputs'][0], p['soft_label'])

            # loss function
            if self.args['kd_ratio'] > 0:
                p['loss'] = (self.model.loss(p['outputs'], p['targets'], self.args['loss_weights']) + self.args['kd_ratio'] * kd_loss) / accum
            else:
                p['loss'] = self.model.loss(p['outputs'], p['targets'], self.args['loss_weights']) / accum
        p['loss'].apply(persistent=True)

        # metrics to monitor during training
        targets = [out.get_unlinked_variable().apply(need_grad=False) for out in p['outputs']]
        p['metrics'] = self.model.metrics(targets, p['targets'])
        for v in p['metrics'].values():
            v.apply(persistent=True)

    def reset_running_statistics(self, setting, net=None, forward_net=None, subset_size=2000, subset_batch_size=200, dataloader=None):
        from ...contrib.classification.ofa.ofa_modules.utils import set_running_statistics

        if net is None:
            net = self.model
        if forward_net is None:
            forward_net = self.forward_model
        if dataloader is None and self.subset_train_dataloader is None:
            self.subset_train_dataloader =\
                self.dataloader['train'].build_sub_train_loader(subset_size, subset_batch_size)
        forward_net.set_active_subnet(**setting)
        set_running_statistics(net, forward_net, self.subset_train_dataloader, subset_size, subset_batch_size, )

    def save_checkpoint(self, path, current_epoch, del_previous=None):
        checkpoint_info = dict()
        checkpoint_file_path = os.path.join(path, 'checkpoint.json')
        saved_path = os.path.join(path, 'saved_weights')
        if self.comm.rank == 0:
            if not os.path.exists(saved_path):
                os.mkdir(saved_path)
        if del_previous is not None and del_previous > 0:
            del_state_path = os.path.join(saved_path, f"states_{current_epoch - del_previous}.h5")
            del_weight_path = os.path.join(saved_path, f"weights_{current_epoch - del_previous}.h5")

            # deleting previous stored states and weights.
            if os.path.exists(del_state_path):
                os.remove(del_state_path)

            if os.path.exists(del_weight_path):
                os.remove(del_weight_path)

        states_path = os.path.join(saved_path, f"states_{current_epoch}.h5")
        self.optimizer['train'].save_states(states_path)
        self.optimizer['train_no_decay'].save_states(states_path)
        checkpoint_info["states_path"] = states_path
        params_path = os.path.join(saved_path, f"weights_{current_epoch}.h5")
        self.model.save_parameters(path=params_path)

        checkpoint_info["params_path"] = params_path
        checkpoint_info["current_epoch"] = current_epoch
        with open(checkpoint_file_path, 'w') as f:
            json.dump(checkpoint_info, f)

    def load_checkpoint(self, path):
        checkoint_file_path = os.path.join(path, f'checkpoint.json')
        print(f"checkoint_file_path : {checkoint_file_path}")
        current_epoch = 0
        if os.path.exists(checkoint_file_path):
            with open(checkoint_file_path, 'r') as f:
                checkpoint_info = json.load(f)
            params_path = checkpoint_info["params_path"]
            self.model.load_parameters(params_path)

            states_path = checkpoint_info["states_path"]
            self.optimizer['train'].load_states(states_path)
            self.optimizer['train_no_decay'].load_states(states_path)
            current_epoch = checkpoint_info["current_epoch"] + 1
        return current_epoch

    def save_model(self, path, epoch=None):

        if self.comm.rank == 0:
            if not os.path.exists(path):
                os.mkdir(path)

            weight_str = "weights"

            # appending the epoch number to the saved weights
            if epoch is not None:
                weight_str += f"_{epoch}"

            weight_str += '.h5'

            self.model.save_parameters(
                path=os.path.join(path, weight_str)
            )

    def callback_on_finish(self):
        pass
