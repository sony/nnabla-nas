# Copyright (c) 2022 Sony Corporation. All Rights Reserved.
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
from nnabla.initializer import ConstantInitializer, UniformInitializer

from ..... import module as Mo
from .....module.initializers import he_initializer, torch_initializer


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
