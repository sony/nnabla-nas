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

import math
import numpy as np

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
from nnabla.initializer import ConstantInitializer, UniformInitializer


from ..... import module as Mo
from .....module.initializers import he_initializer, torch_initializer
from ..ofa_modules.dynamic_op import DynamicBatchNorm2d
from .my_random_resize_crop import MyResize


def init_models(net, model_init='he_fout'):
    """ Initilizes parameters of Convolution, BatchNormalization, Linear,"""
    if isinstance(net, list):
        for sub_net in net:
            init_models(sub_net, model_init)
        return
    for _, m in net.get_modules():
        if isinstance(m, Mo.Conv):
            if model_init == 'he_fout':
                w_init = he_initializer(m._out_channels, m._kernel[0], rng=None)
                m._W = Mo.Parameter(
                    m._W.shape, initializer=w_init, scope=m._scope_name)
            elif model_init == 'he_fin':
                w_init = he_initializer(m._in_channels, m._kernel[0], rng=None)
                m._W = Mo.Parameter(
                    m._W.shape, initializer=w_init, scope=m._scope_name)
            elif model_init == 'pytorch':
                w_init = torch_initializer(
                    m._in_channels, m._kernel, rng=None)
                m._W = Mo.Parameter(
                    m._W.shape, initializer=w_init, scope=m._scope_name)
            else:
                raise NotImplementedError
            if m._b is not None:
                b_init = ConstantInitializer(0)
                m._b = Mo.Parameter(
                    m._b.shape, initializer=b_init, scope=m._scope_name)
        elif isinstance(m, Mo.BatchNormalization):
            beta_init = ConstantInitializer(0)
            m._beta = Mo.Parameter(
                m._beta.shape, initializer=beta_init, scope=m._scope_name)
            gamma_init = ConstantInitializer(1)
            m._gamma = Mo.Parameter(
                m._gamma.shape, initializer=gamma_init, scope=m._scope_name)
        elif isinstance(m, Mo.Linear):
            stdv = 1. / math.sqrt(m._W.shape[1])
            w_init = UniformInitializer((-stdv, stdv))
            m._W = Mo.Parameter(
                m._W.shape, initializer=w_init, scope=m._scope_name)
            if m._b is not None:
                b_init = ConstantInitializer(0)
                m._b = Mo.Parameter(
                    m._b.shape, initializer=b_init, scope=m._scope_name)


def set_running_statistics(model, dataloader, dataloader_batch_size, data_size, batch_size, inp_shape):
    """Re-calculates batch normalization mean and variance"""

    logger.info('Start batch normalization stats calibaration')

    model.apply(training=False)

    for name, m in model.get_modules():
        if isinstance(m, Mo.BatchNormalization):
            m.training = True
            m.set_running_statistics = True
            m.mean_est.reset()
            m.var_est.reset()
        elif isinstance(m, DynamicBatchNorm2d):
            m.set_running_statistics = True

    def load_data(placeholder, data):
        inp_list = []
        for inp, x in zip(placeholder, data):
            if isinstance(x, nn.NdArray):
                inp.data = x
            else:
                inp.d = x
            inp_list.append(inp)
        return inp_list

    with nn.no_grad():
        resize = MyResize()
        transform = dataloader.transform('valid')
        with nn.auto_forward(True):
            # Note: only support NCHW data format
            x = [nn.Variable([dataloader_batch_size] + shape) for shape in inp_shape]
            accum = batch_size // dataloader_batch_size + 1
            for i in range(data_size // batch_size):
                x_accum = []
                for _ in range(accum):
                    data = dataloader.next()
                    x_accum.append(load_data(x, data['inputs']))
                x_concat = [F.concatenate(*[x_accum[i][j] for i in range(len(x_accum))], axis=0)
                            for j in range(len(x))]
                inputs = [resize(transform(x[:batch_size, :, :, :])) for x in x_concat]
                model(*inputs)

    for name, m in model.get_modules():
        if isinstance(m, Mo.BatchNormalization):
            if m.mean_est.count > 0:
                feature_dim = m.mean_est.avg.shape[1]
                new_mean = np.concatenate(
                    (m.mean_est.avg, m._mean.d[:, feature_dim:, :, :]), axis=1)
                new_var = np.concatenate(
                    (m.var_est.avg, m._var.d[:, feature_dim:, :, :]), axis=1)
                m._mean.d = new_mean
                m._var.d = new_var
            m.set_running_statistics = False
        elif isinstance(m, DynamicBatchNorm2d):
            m.set_running_statistics = False

    logger.info('Batch normalization stats calibaration finished')