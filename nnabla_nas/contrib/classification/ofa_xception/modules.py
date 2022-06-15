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

from collections import OrderedDict

from .... import module as Mo
from ..ofa.ofa_utils.common_tools import get_same_padding, min_divisible_value

import nnabla as nn
import nnabla.functions as F
from nnabla.initializer import ConstantInitializer


CANDIDATES = {
    # we'll need 27 options
    'XP1 3x3 1': {'ks': 3, 'depth': 1, 'expand_ratio': 1},
    'XP1 3x3 2': {'ks': 3, 'depth': 2, 'expand_ratio': 1},
    'XP1 3x3 3': {'ks': 3, 'depth': 3, 'expand_ratio': 1},
    'XP0.8 3x3 1': {'ks': 3, 'depth': 1, 'expand_ratio': 0.8},
    'XP0.8 3x3 2': {'ks': 3, 'depth': 2, 'expand_ratio': 0.8},
    'XP0.8 3x3 3': {'ks': 3, 'depth': 3, 'expand_ratio': 0.8},
    'XP0.6 3x3 1': {'ks': 3, 'depth': 1, 'expand_ratio': 0.6},
    'XP0.6 3x3 2': {'ks': 3, 'depth': 2, 'expand_ratio': 0.6},
    'XP0.6 3x3 3': {'ks': 3, 'depth': 3, 'expand_ratio': 0.6},

    'XP1 5x5 1': {'ks': 5, 'depth': 1, 'expand_ratio': 1},
    'XP1 5x5 2': {'ks': 5, 'depth': 2, 'expand_ratio': 1},
    'XP1 5x5 3': {'ks': 5, 'depth': 3, 'expand_ratio': 1},
    'XP0.8 5x5 1': {'ks': 5, 'depth': 1, 'expand_ratio': 0.8},
    'XP0.8 5x5 2': {'ks': 5, 'depth': 2, 'expand_ratio': 0.8},
    'XP0.8 5x5 3': {'ks': 5, 'depth': 3, 'expand_ratio': 0.8},
    'XP0.6 5x5 1': {'ks': 5, 'depth': 1, 'expand_ratio': 0.6},
    'XP0.6 5x5 2': {'ks': 5, 'depth': 2, 'expand_ratio': 0.6},
    'XP0.6 5x5 3': {'ks': 5, 'depth': 3, 'expand_ratio': 0.6},

    'XP1 7x7 1': {'ks': 7, 'depth': 1, 'expand_ratio': 1},
    'XP1 7x7 2': {'ks': 7, 'depth': 2, 'expand_ratio': 1},
    'XP1 7x7 3': {'ks': 7, 'depth': 3, 'expand_ratio': 1},
    'XP0.8 7x7 1': {'ks': 7, 'depth': 1, 'expand_ratio': 0.8},
    'XP0.8 7x7 2': {'ks': 7, 'depth': 2, 'expand_ratio': 0.8},
    'XP0.8 7x7 3': {'ks': 7, 'depth': 3, 'expand_ratio': 0.8},
    'XP0.6 7x7 1': {'ks': 7, 'depth': 1, 'expand_ratio': 0.6},
    'XP0.6 7x7 2': {'ks': 7, 'depth': 2, 'expand_ratio': 0.6},
    'XP0.6 7x7 3': {'ks': 7, 'depth': 3, 'expand_ratio': 0.6},

    'skip_connect': {'ks': None, 'depth': None, 'expand_ratio': None},
}


def candidates2subnetlist(candidates):
    ks_list = []
    expand_list = []
    for candidate in candidates:
        ks = CANDIDATES[candidate]['ks']
        e = CANDIDATES[candidate]['expand_ratio']
        if ks not in ks_list:
            ks_list.append(ks)
        if e not in expand_list:
            expand_list.append(e)
    return ks_list, expand_list


def genotype2subnetlist(op_candidates, genotype):
    op_candidates.append('skip_connect')
    subnet_list = [op_candidates[i] for i in genotype]
    ks_list = [CANDIDATES[subnet]['ks'] if subnet != 'skip_connect'
               else 3 for subnet in subnet_list]
    expand_ratio_list = [CANDIDATES[subnet]['expand_ratio'] if subnet != 'skip_connect'
                         else 4 for subnet in subnet_list]
    depth_list = [CANDIDATES[subnet]['depth'] if subnet != 'skip_connect' else 3 for subnet in subnet_list]
    assert([d >= 1 for d in depth_list])
    return ks_list, expand_ratio_list, depth_list


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        LinearLayer.__name__: LinearLayer,
        XceptionBlock.__name__: XceptionBlock,
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


def force_tuple2(value):
    if value is None:
        return value
    if hasattr(value, '__len__'):
        assert len(value) == 2
        return value
    return (value,) * 2


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

    def build_from_config(config):
        return ConvLayer(**config)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'dilation={self._dilation}, '
                f'group={self._group}, '
                f'with_bias={self._with_bias}, '
                f'use_bn={self._use_bn}, '
                f'act_func={self._act_func}, '
                f'name={self._name}')


class DwConvLayer(Mo.Sequential):

    r""" Depthwise Convolution-BatchNormalization(optional)-Activation layer.

    Args:
        in_channels (int): Number of convolution kernels (which is
            equal to the number of input channels).
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

    def __init__(self, in_channels, kernel=(3, 3),
                 stride=(1, 1), dilation=(1, 1), with_bias=False,
                 use_bn=True, act_func='relu'):
        self._in_channels = in_channels
        self._kernel = kernel
        self._stride = stride
        self._dilation = dilation
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
        module_dict['conv'] = Mo.DwConv(self._in_channels, self._kernel,
                                        pad=padding, stride=self._stride, dilation=self._dilation,
                                        with_bias=self._with_bias)
        if self._use_bn:
            module_dict['bn'] = Mo.BatchNormalization(self._in_channels, 4)
        module_dict['act'] = build_activation(act_func)

        super(DwConvLayer, self).__init__(module_dict)

    def build_from_config(config):
        return DwConvLayer(**config)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'dilation={self._dilation}, '
                f'with_bias={self._with_bias}, '
                f'use_bn={self._use_bn}, '
                f'act_func={self._act_func}, '
                f'name={self._name}')


class SeparableConv(Mo.Module):
    def __init__(
            self, in_channels, out_channels, kernel=(1, 1),
            stride=(1, 1),
            pad=(0, 0),
            dilation=(1, 1),
            use_bn=True, act_fn=None):
        super(SeparableConv, self).__init__()

        self.conv1 = Mo.Conv(in_channels, in_channels, kernel, pad=pad, dilation=dilation,
                             stride=stride, with_bias=False, group=in_channels)
        self.pointwise = Mo.Conv(
            in_channels, out_channels, (1, 1),
            stride=(1, 1),
            pad=(0, 0),
            dilation=(1, 1),
            group=1, with_bias=False)

        self.use_bn = use_bn
        if use_bn:
            self.bn = Mo.BatchNormalization(out_channels, 4)
        self.act = build_activation(act_fn)

    def call(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)

        if self.use_bn:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x

    @staticmethod
    def build_from_config(config):
        return SeparableConv(**config)


class XceptionBlock(Mo.Module):
    def __init__(
            self, in_channels, out_channels, reps, kernel=(3, 3),
            stride=(1, 1),
            start_with_relu=True, grow_first=True, expand_ratio=None, mid_channels=None):
        super(XceptionBlock, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        if out_channels != in_channels or stride != 1:
            self.skip = Mo.Conv(in_channels, out_channels, (1, 1), stride=stride, with_bias=False)
            self.skipbn = Mo.BatchNormalization(out_channels, 4)
        else:
            self.skip = None

        rep = []
        if expand_ratio is None:
            for i in range(reps):
                if grow_first:
                    inc = in_channels if i == 0 else out_channels
                    outc = out_channels
                else:
                    inc = in_channels
                    outc = in_channels if i < (reps - 1) else out_channels
                rep.append(Mo.ReLU(inplace=True))
                rep.append(SeparableConv(inc, outc, kernel=kernel, stride=(1, 1), pad=(1, 1)))
        elif reps == 3:
            rep.append(Mo.ReLU(inplace=True))
            rep.append(SeparableConv(in_channels, mid_channels, kernel=kernel, stride=(1, 1), pad=(1, 1)))

            rep.append(Mo.ReLU(inplace=True))
            rep.append(SeparableConv(mid_channels, mid_channels, kernel=kernel, stride=(1, 1), pad=(1, 1)))

            rep.append(Mo.ReLU(inplace=True))
            rep.append(SeparableConv(mid_channels, out_channels, kernel=kernel, stride=(1, 1), pad=(1, 1)))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = Mo.ReLU(inplace=False)

        if stride != 1:
            rep.append(Mo.MaxPool((3, 3), stride=stride, pad=(1, 1)))
        self.rep = Mo.Sequential(*rep)

    def call(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

    @staticmethod
    def build_from_config(config):
        return XceptionBlock(**config)


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
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'bias={self._bias}, '
                f'drop_rate={self._drop_rate}, '
                f'name={self._name} ')


class FusedBatchNormalization(Mo.Module):
    def __init__(self, n_features, n_dims, z=None, axes=[1], decay_rate=0.9, eps=1e-5,
                 nonlinearity='relu', output_stat=False, fix_parameters=False, param_init=None,
                 name=''):
        Mo.Module.__init__(self, name=name)
        self._scope_name = f'<fusedbatchnorm at {hex(id(self))}>'

        assert len(axes) == 1

        shape_stat = [1 for _ in range(n_dims)]
        shape_stat[axes[0]] = n_features

        if param_init is None:
            param_init = {}
        beta_init = param_init.get('beta', ConstantInitializer(0))
        gamma_init = param_init.get('gamma', ConstantInitializer(1))
        mean_init = param_init.get('mean', ConstantInitializer(0))
        var_init = param_init.get('var', ConstantInitializer(1))

        if fix_parameters:
            self._beta = nn.Variable.from_numpy_array(
                beta_init(shape_stat))
            self._gamma = nn.Variable.from_numpy_array(
                gamma_init(shape_stat))
        else:
            self._beta = Mo.Parameter(shape_stat, initializer=beta_init,
                                      scope=self._scope_name)
            self._gamma = Mo.Parameter(shape_stat, initializer=gamma_init,
                                       scope=self._scope_name)

        self._mean = Mo.Parameter(shape_stat, need_grad=False,
                                  initializer=mean_init,
                                  scope=self._scope_name)
        self._var = Mo.Parameter(shape_stat, need_grad=False,
                                 initializer=var_init,
                                 scope=self._scope_name)
        self._z = z
        self._axes = axes
        self._decay_rate = decay_rate
        self._eps = eps
        self._n_features = n_features
        self._fix_parameters = fix_parameters
        self._output_stat = output_stat
        self._nonlinearity = nonlinearity

    def call(self, input):
        return F.fused_batch_normalization(input, self._beta, self._gamma,
                                           self._mean, self._var, self._z, self._axes,
                                           self._decay_rate, self._eps,
                                           self.training, self._nonlinearity, self._output_stat)

    def extra_repr(self):
        return (f'n_features={self._n_features}, '
                f'fix_parameters={self._fix_parameters}, '
                f'eps={self._eps}, '
                f'decay_rate={self._decay_rate}')

    @staticmethod
    def build_from_config(config):
        return FusedBatchNormalization(**config)


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
