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
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
from nnabla.initializer import ConstantInitializer, UniformInitializer

from ..... import module as Mo
from ...base import ClassificationModel as Model
from .....utils.initializers import he_initializer
from .common_tools import cross_entropy_loss_with_label_smoothing


def set_bn_param(net, decay_rate, eps, gn_channel_per_group, ws_eps=None, **kwargs):
    for _, m in net.get_modules():
        if isinstance(m, Mo.BatchNormalization):
            m._decay_rate = decay_rate
            m._eps = eps


def get_bn_param(net):
    for _, m in net.get_modules():
        if isinstance(m, Mo.BatchNormalization):
            return {
                'decay_rate': m._decay_rate,
                'eps': m._eps
            }


def init_models(net, model_init='he_fout'):
    """
        Conv2d,
        BatchNorm2d, BatchNorm1d, GroupNorm
        Linear,
    """
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


class MyGlobalAvgPool2d(Mo.Module):
    r"""Global average pooling layer.
    It pools an averaged value from the whole image.
    Args:
        name (string): the name of this module
    """
    def __init__(self, keep_dims=True, name=''):
        self._name = name
        self._scope_name = f'<myglobalavgpool at {hex(id(self))}>'
        self._keep_dims = keep_dims

    def call(self, input):
        return F.mean(input, axis=(2, 3), keepdims=self._keep_dims)


class MyConv2d(Mo.Conv):
    def __init__(self, in_channels, out_channels, kernel, pad=None,
                 stride=None, dilation=None, group=1, w_init=None, b_init=None,
                 base_axis=1, fix_parameters=False, rng=None, with_bias=True,
                 channel_last=False, name=''):
        super(MyConv2d, self).__init__(in_channels, out_channels, kernel, pad=None,
                                       stride=None, dilation=None, group=1, w_init=None, b_init=None,
                                       base_axis=1, fix_parameters=False, rng=None, with_bias=True,
                                       channel_last=False, name='')
        self._scope_name = f'<myconv2d at {hex(id(self))}>'
        self.WS_EPS = None

    def call(self, input):
        if self.WS_EPS is None:
            return super(MyConv2d, self).forward()
        else:
            return F.convolution(input, F.weight_standardozation(self._W), self._b, self._base_axis,
                                 self._pad, self._stride, self._dilation,
                                 self._group, self._channel_last)


class MyNetwork(Model):
    CHANNEL_DIVISIBLE = 8

    def set_bn_param(self, decay_rate, eps, gn_channel_per_group=None, **kwargs):
        set_bn_param(self, decay_rate, eps, gn_channel_per_group, **kwargs)

    def get_bn_param(self):
        return get_bn_param(self)

    def loss(self, outputs, targets, loss_weights=None):
        return cross_entropy_loss_with_label_smoothing(outputs[0], targets[0])

    def get_net_parameters(self, keys=None, mode='include', grad_only=False):
        r"""Returns an `OrderedDict` containing model parameters.

        Args:
            grad_only (bool, optional): If sets to `True`, then only parameters
                with `need_grad=True` are returned. Defaults to False.

        Returns:
            OrderedDict: A dictionary containing parameters.
        """
        p = self.get_parameters(grad_only)
        if keys is None:
            return OrderedDict([(k, v) for k, v in p.items() if 'alpha' not in k])
        elif mode == 'include':  # without weight decay
            param_dict = OrderedDict()
            for name, param in p.items():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag:
                    param_dict[name] = param
            return param_dict
        elif mode == 'exclude':  # with weight decay
            param_dict = OrderedDict()
            for name, param in p.items():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag:
                    param_dict[name] = param
            return param_dict
        else:
            raise ValueError('do not support %s' % mode)

    def load_ofa_parameters(self, path, raise_if_missing=False):
        with nn.parameter_scope('', OrderedDict()):
            nn.load_parameters(path)
            params = nn.get_parameters(grad_only=False)
        self.set_ofa_parameters(params, raise_if_missing=raise_if_missing)

    def set_ofa_parameters(self, params, raise_if_missing=False):
        for prefix, module in self.get_modules():
            for name, p in module.parameters.items():
                key = prefix + ('/' if prefix else '') + name
                if '/mobile_inverted_conv/' in key:
                    new_key = key.replace('/mobile_inverted_conv/', '/conv/')
                else:
                    new_key = key
                if new_key in params:
                    pass
                else:
                    if raise_if_missing:
                        raise ValueError(
                            f'A child module {name} cannot be found in '
                            '{this}. This error is raised because '
                            '`raise_if_missing` is specified '
                            'as True. Please turn off if you allow it.')
                p.d = params[new_key].d.copy()
                nn.logger.info(f'`{new_key}` loaded.')
