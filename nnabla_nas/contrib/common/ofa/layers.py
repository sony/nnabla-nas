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
from collections import OrderedDict
import nnabla.functions as F
from nnabla.initializer import ConstantInitializer, UniformInitializer

from .... import module as Mo
from .utils.common_tools import get_same_padding, min_divisible_value, make_divisible
from ....module.initializers import he_initializer, torch_initializer


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        LinearLayer.__name__: LinearLayer,
        MBConvLayer.__name__: MBConvLayer,
        'MBInvertedConvLayer': MBConvLayer,
        ##########################################################
        ResidualBlock.__name__: ResidualBlock,
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


def set_bn_param(net, decay_rate, eps, **kwargs):
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


def get_extra_repr(cur_obj):
    repr = ""
    for var in vars(cur_obj):
        # Skip this field just for a better printout
        if var == '_modules':
            continue

        var_value = getattr(cur_obj, var)
        repr += f'{var}='
        repr += f'{var_value}, '

    repr += ')'
    return repr


class ResidualBlock(Mo.Module):
    r"""ResidualBlock layer.

    Adds outputs of a convolution layer and a shortcut.

    Args:
        conv (:obj:`Module`): A convolution module.
        shortcut (:obj:`Module`): An identity module.
    """

    def __init__(self, conv, shortcut):
        self.conv = conv
        self.shortcut = shortcut

    def call(self, x):
        if self.conv is None:
            res = x
        elif self.shortcut is None:
            res = self.conv(x)
        else:
            res = self.conv(x) + self.shortcut(x)
        return res

    @staticmethod
    def build_from_config(config):
        conv_config = config['conv'] if 'conv' in config else config['mobile_inverted_conv']
        conv = set_layer_from_config(conv_config)
        shortcut = Mo.Identity()
        return ResidualBlock(conv, shortcut)


class ConvLayer(Mo.Sequential):

    r"""Convolution-BatchNormalization(optional)-Activation layer.

    Args:
        in_channels (int): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (int): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        kernel (tuple of int, optional): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3, 5). Defaults to (3, 3)
        stride (tuple of int, optional): Stride sizes for
            dimensions. Defaults to (1, 1).
        dilation (tuple of int, optional): Dilation sizes for
            dimensions. Defaults to (1, 1).
        group (int, optional): Number of groups of channels.
            Defaults to 1.
        with_bias (bool, optional): If True, bias for Convolution is added.
            Defaults to False.
        use_bn (bool, optional): If True, BatchNormalization layer is added.
            Defaults to True.
        act_func (str, optional) Type of activation. Defaults to 'relu'.
    """

    def __init__(self, in_channels, out_channels, kernel=(3, 3),
                 stride=(1, 1), dilation=(1, 1), group=1, with_bias=False,
                 use_bn=True, act_func='relu'):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._stride = stride
        self._dilation = dilation
        self._group = group
        self._with_bias = with_bias
        self._use_bn = use_bn
        self._act_func = act_func

        padding = get_same_padding(self._kernel)
        if isinstance(padding, int):
            padding *= self._dilation
        else:
            new_padding = (padding[0] * self._dilation[0], padding[1] * self._dilation[1])
            padding = tuple(new_padding)

        module_dict = OrderedDict()
        module_dict['conv'] = Mo.Conv(self._in_channels, self._out_channels, self._kernel,
                                      pad=padding, stride=self._stride, dilation=self._dilation,
                                      group=min_divisible_value(self._in_channels, self._group),
                                      with_bias=self._with_bias)
        if self._use_bn:
            module_dict['bn'] = Mo.BatchNormalization(out_channels, 4)
        module_dict['act'] = build_activation(act_func)

        super(ConvLayer, self).__init__(module_dict)

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)

    def extra_repr(self):
        return get_extra_repr(self)


class MBConvLayer(Mo.Module):

    r"""The inverted layer with optional squeeze-and-excitation.

    Args:
        in_channels (int): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (int): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        kernel (tuple of int): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3, 5). Defaults to (3, 3)
        stride (tuple of int, optional): Stride sizes for dimensions.
            Defaults to (1, 1).
        expand_ratio (int): The expand ratio.
        mid_channels (int): The number of features. Defaults to None.
        act_func (str) Type of activation. Defaults to 'relu'.
        use_se (bool, optional): If True, squeeze-and-expand module is used.
            Defaults to False.
        group (int, optional): Number of groups of channels.
            Defaults to 1.
    """

    def __init__(self, in_channels, out_channels,
                 kernel=(3, 3), stride=(1, 1), expand_ratio=6, mid_channels=None,
                 act_func='relu6', use_se=False, group=None):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._stride = stride
        self._expand_ratio = expand_ratio
        self._mid_channels = mid_channels
        self._act_func = act_func
        self._use_se = use_se
        self._group = group

        if self._mid_channels is None:
            feature_dim = round(self._in_channels * self._expand_ratio)
        else:
            feature_dim = self._mid_channels

        if self._expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = Mo.Sequential(OrderedDict([
                ('conv', Mo.Conv(
                    self._in_channels, feature_dim, (1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)),
                ('bn', Mo.BatchNormalization(feature_dim, 4)),
                ('act', build_activation(self._act_func, inplace=True))
            ]))

        pad = get_same_padding(self._kernel)
        group = feature_dim if self._group is None else min_divisible_value(feature_dim, self._group)
        depth_conv_modules = [
            ('conv', Mo.Conv(
                feature_dim, feature_dim, kernel, pad=pad, stride=stride, group=group, with_bias=False)),
            ('bn', Mo.BatchNormalization(feature_dim, 4)),
            ('act', build_activation(self._act_func, inplace=True)),
        ]
        if self._use_se:
            depth_conv_modules.append(('se', SEModule(feature_dim)))
        self.depth_conv = Mo.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = Mo.Sequential(OrderedDict([
            ('conv', Mo.Conv(feature_dim, out_channels, (1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)),
            ('bn', Mo.BatchNormalization(out_channels, 4))
        ]))

    def call(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @staticmethod
    def build_from_config(config):
        return MBConvLayer(**config)

    def extra_repr(self):
        return get_extra_repr(self)


class LinearLayer(Mo.Sequential):

    r"""Affine, or fully connected layer with dropout.

    Args:
        in_features (int): The size of each input sample.
        in_features (int): The size of each output sample.
        with_bias (bool): Specify whether to include the bias term.
            Defaults to True.
        drop_rate (float, optional): Dropout ratio applied to parameters.
            Defaults to 0.
    """

    def __init__(self, in_features, out_features, bias=True, drop_rate=0):
        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias
        self._drop_rate = drop_rate

        super(LinearLayer, self).__init__(OrderedDict({
            'dropout': Mo.Dropout(self._drop_rate),
            'linear': Mo.Linear(self._in_features, self._out_features, bias=self._bias),
        }))

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)

    def extra_repr(self):
        return get_extra_repr(self)


class SEModule(Mo.Module):

    r"""Squeeze-and-Excitation module, that adaptively recalibrates channel-wise
        feature responces by explicitlly modelling interdependencies
        between channels.

    Args:
        channel (int): The number of input channels.
        reduction (int, optional): The reduction rate used to determine
            the number of middle channels.

    References:
    [1] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks."
        Proceedings of the IEEE conference on computer vision and pattern
        recognition. 2018.
    """

    REDUCTION = 4

    def __init__(self, channel, reduction=None, name=''):
        self._name = name
        self._scope_name = f'<semodule at {hex(id(self))}>'

        self._channel = channel
        self.reduction = SEModule.REDUCTION if reduction is None else reduction

        num_mid = make_divisible(self._channel // self.reduction)

        self.fc = Mo.Sequential(OrderedDict([
            ('reduce', Mo.Conv(
                self._channel, num_mid, (1, 1), pad=(0, 0), stride=(1, 1), with_bias=True)),
            ('relu', Mo.ReLU()),
            ('expand', Mo.Conv(
                num_mid, self._channel, (1, 1), pad=(0, 0), stride=(1, 1), with_bias=True)),
            ('h_sigmoid', Mo.Hsigmoid())
        ]))

    def call(self, input):
        y = F.mean(input, axis=(2, 3), keepdims=True)
        y = self.fc(y)
        return input * y


def build_activation(act_func, inplace=False):
    if act_func == 'relu':
        return Mo.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return Mo.ReLU6()
    elif act_func == 'h_swish':
        return Mo.Hswish()
    elif act_func is None or act_func == 'none':
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


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
