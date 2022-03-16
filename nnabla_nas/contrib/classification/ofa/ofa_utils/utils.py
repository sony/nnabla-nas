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
from .....utils.helper import AverageMeter
from .....module.initializers import he_initializer
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
                m._W = Mo.Parameter(m._W.shape, initializer=w_init, scope=m._scope_name)
            elif model_init == 'he_fin':
                he_init = he_initializer(m._in_channels, m._kernel[0], rng=None)
                m._W = Mo.Parameter(m._W.shape, initializer=he_init, scope=m._scope_name)
            else:
                raise NotImplementedError
            if m._b is not None:
                b_init = ConstantInitializer(0)
                m._b = Mo.Parameter(m._b.shape, initializer=b_init, scope=m._scope_name)
        elif isinstance(m, Mo.BatchNormalization):
            beta_init = ConstantInitializer(0)
            m._beta = Mo.Parameter(m._beta.shape, initializer=beta_init, scope=m._scope_name)
            gamma_init = ConstantInitializer(1)
            m._gamma = Mo.Parameter(m._gamma.shape, initializer=gamma_init, scope=m._scope_name)
        elif isinstance(m, Mo.Linear):
            stdv = 1. / math.sqrt(m._W.shape[1])
            w_init = UniformInitializer((-stdv, stdv))
            m._W = Mo.Parameter(m._W.shape, initializer=w_init, scope=m._scope_name)
            if m._b is not None:
                b_init = ConstantInitializer(0)
                m._b = Mo.Parameter(m._b.shape, initializer=b_init, scope=m._scope_name)


def set_running_statistics(model, dataloader, dataloader_batch_size, data_size, batch_size):
    """Re-calculates batch normalization mean and variance"""

    logger.info('Start batch normalization stats calibaration')

    model.apply(training=False)

    bn_mean = {}
    bn_var = {}
    bn_params = {}
    for name, m in model.get_modules():
        if isinstance(m, Mo.BatchNormalization):
            bn_params[name] = {
                'beta': m._beta, 'gamma': m._gamma, 'mean': m._mean, 'variance': m._var,
                'axes': m._axes, 'decay_rate': m._decay_rate, 'eps': m._eps,
                'batch_stat': m._training, 'output_stat': m._output_stat,
            }
            m.training = True
            bn_mean[name] = AverageMeter(name)
            bn_var[name] = AverageMeter(name)

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = F.mean(x, axis=(0, 2, 3), keepdims=True)
                    batch_var = F.mean(x ** 2, axis=(0, 2, 3), keepdims=True) - batch_mean ** 2

                    mean_est.update(batch_mean.d, x.shape[0])
                    var_est.update(batch_var.d, x.shape[0])

                    _feature_dim = batch_mean.shape[1]
                    return F.batch_normalization(
                        x, bn._beta[:, :_feature_dim, :, :], bn._gamma[:, :_feature_dim, :, :],
                        batch_mean, batch_var, decay_rate=1, eps=1e-5, batch_stat=False
                    )
                return lambda_forward

            m.call = new_forward(m, bn_mean[name], bn_var[name])

    def load_data(inp, x):
        if isinstance(x, nn.NdArray):
            inp.data = x
        else:
            inp.d = x
        return inp

    with nn.no_grad():
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
        with nn.auto_forward(True):
            transform = MyResize()
            x = nn.Variable(shape=(dataloader_batch_size, 3, 224, 224))
            accum = batch_size // dataloader_batch_size + 1
            for i in range(data_size // batch_size):
                x_accum = []
                for _ in range(accum):
                    data = dataloader.next()
                    x_accum.append(load_data(x, data['inputs'][0]))
                x_accum = F.concatenate(*x_accum, axis=0)
                model(*[transform(x_accum[:batch_size, :, :, :])])
            DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

    for name, m in model.get_modules():
        if isinstance(m, Mo.BatchNormalization):
            if name in bn_mean and bn_mean[name].count > 0:
                feature_dim = bn_mean[name].avg.shape[1]
                new_mean = np.concatenate((bn_mean[name].avg, m._mean.d[:, feature_dim:, :, :]), axis=1)
                new_var = np.concatenate((bn_var[name].avg, m._var.d[:, feature_dim:, :, :]), axis=1)
                m._mean.d = new_mean
                m._var.d = new_var

            def new_forward(bn, params):
                def lambda_forward(x):
                    return F.batch_normalization(x, **params)
                return lambda_forward

            m.call = new_forward(m, bn_params[name])

    logger.info('Batch normalization stats calibaration finished')
