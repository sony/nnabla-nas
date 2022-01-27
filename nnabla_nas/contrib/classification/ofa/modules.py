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

import copy
from collections import OrderedDict

import numpy as np

import nnabla as nn
import nnabla.functions as F

from .... import module as Mo
from .ofa_modules.common_tools import make_divisible, get_same_padding, min_divisible_value
from .ofa_modules.pytorch_modules import SEModule, Hswish
from .ofa_modules.my_modules import MyNetwork


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        MBConvLayer.__name__: MBConvLayer,
        'MBInvertedConvLayer': MBConvLayer,
        ##########################################################
        ResidualBlock.__name__: ResidualBlock,
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class ResidualBlock(Mo.Module):
    def __init__(self, conv, shortcut):
        super(ResidualBlock, self).__init__()
        self._scope_name = f'<residualblock at {hex(id(self))}>'

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

    @property
    def module_str(self):
        return '(%s, %s)' % (self.conv.module_str, 'shortcut')

    """@property
    def get_active_parameter_size(self):
        param_W = np.prod(self.conv.conv._W.shape)
        param_b = np.prod(self.conv.conv._b.shape)
        return param_W + param_b"""

    @staticmethod
    def build_from_config(config):
        conv_config = config['conv'] if 'conv' in config else config['mobile_inverted_conv']
        conv = set_layer_from_config(conv_config)
        shortcut = set_layer_from_config(config['shortcut'])
        return ResidualBlock(conv, shortcut)


class My2DLayer(Mo.Module):

    def __init__(self, in_channels, out_channels,
                 use_bn=True, use_se=False, act_func='relu', dropout=0, ops_order='weight_bn_act'):

        super(My2DLayer, self).__init__()

        self._scope_name = f'<my2dlayer at {hex(id(self))}>'
        self._in_channels = in_channels
        self._out_channels = out_channels

        self._use_bn = use_bn
        self._use_se = use_se
        self._act_func = act_func
        self._dropout = dropout
        self._ops_order = ops_order

        modules = {}
        # bn
        if self._use_bn:
            if self.bn_before_weight:
                modules['bn'] = Mo.BatchNormalization(in_channels, 4)
            else:
                modules['bn'] = Mo.BatchNormalization(out_channels, 4)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self._act_func)
        # dropout
        if self._dropout > 0:
            modules['dropout'] = Mo.Dropout(self._dropout)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        module_dict = OrderedDict()
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is None:
                    module_dict['dropout'] = modules['dropout']
                for key in modules['weight']:
                    module_dict[key] = modules['weight'][key]
            else:
                module_dict[op] = modules[op]
        if self._use_se:
            module_dict['se'] = SEModule(self._out_channels)
        self._layer = Mo.Sequential(module_dict)

    @property
    def ops_list(self):
        return self._ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self._ops_order)

    def weight_op(self):
        raise NotImplementedError

    def call(self, input):
        for module in self._layer:
            input = module(input)
        return input

    @property
    def module_str(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class ConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel=(3, 3), stride=(1, 1),
                 dilation=(1, 1), group=1, with_bias=False,
                 has_shuffle=False, use_se=False, use_bn=True, act_func='relu',
                 dropout=0, ops_order='weight_bn_act'):

        self._scope_name = f'<convlayer at {hex(id(self))}>'

        self._kernel = kernel
        self._stride = stride
        self._dilation = dilation
        self._group = group
        self._with_bias = with_bias
        self._has_shuffle = has_shuffle
        self._use_se = use_se

        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, use_se, act_func, dropout, ops_order)

    def weight_op(self):
        padding = get_same_padding(self._kernel)
        if isinstance(padding, int):
            padding *= self._dilation
        else:
            new_padding = (padding[0] * self._dilation[0], padding[1] * self._dilation[1])
            padding = tuple(new_padding)

        weight_dict = OrderedDict({
            'conv': Mo.Conv(
                self._in_channels, self._out_channels, self._kernel, pad=padding,
                stride=self._stride, dilation=self._dilation,
                group=min_divisible_value(self._in_channels, self._group),
                with_bias=self._with_bias)
        })
        if self._has_shuffle and self._group > 1:
            print('shuffle not implemented yet')

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                conv_str = '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                conv_str = '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                conv_str = '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])
        conv_str += '_O%d' % self.out_channels
        if self.use_se:
            conv_str = 'SE_' + conv_str
        conv_str += '_' + self.act_func.upper()
        if self.use_bn:
            if isinstance(self.bn, nn.GroupNorm):
                conv_str += '_GN%d' % self.bn.num_groups
            elif isinstance(self.bn, nn.BatchNorm2d):
                conv_str += '_BN'
        return conv_str

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)


class MBConvLayer(Mo.Module):
    def __init__(self, in_channels, out_channels,
                 kernel=(3, 3), stride=(1, 1), expand_ratio=6, mid_channels=None,
                 act_func='relu6', use_se=False, group=None):

        super(MBConvLayer, self).__init__()
        self._scope_name = f'<mbconvlayer at {hex(id(self))}>'

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

    def call(self, input):
        if self.inverted_bottleneck:
            input = self.inverted_bottleneck(input)
        x = self.depth_conv(input)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self._mid_channels is None:
            expand_ratio = self._expand_ratio
        else:
            expand_ratio = self._mid_channels // self._in_channels
        layer_str = '%dx%d_MBConv%d_%s' % (self._kernel, self._kernel, expand_ratio, self.act_func.upper())
        if self._use_se:
            layer_str = 'SE_' + layer_str
        layer_str += '_O%d' % self._out_channels
        if self._group is not None:
            layer_str += '_G%d' % self._group
        if isinstance(self.point_linear.bn, nn.GroupNorm):
            layer_str += '_GN%d' % self.point_linear.bn.num_groups
        elif isinstance(self.point_linear.bn, nn.BatchNorm2d):
            layer_str += '_BN'

        return layer_str

    @staticmethod
    def build_from_config(config):
        return MBConvLayer(**config)


class ResNetBottleneckBlock(Mo.Module):
    def __init__(self, in_channels, out_channels,
                 kernel=(3, 3), stride=(1, 1), expand_ratio=0.25, mid_channels=None, act_func='relu', group=1,
                 downsample_mode='avgpool_conv'):

        super(ResNetBottleneckBlock, self).__init__()
        self._scope_name = f'<resnetbottleneclblock at {hex(id(self))}>'
        self._in_channels = in_channels
        self._out_channels = out_channels

        self._kernel = kernel
        self._stride = stride
        self._expand_ratio = expand_ratio
        self._mid_channels = mid_channels
        self._act_func = act_func
        self._group = group

        self._downsample_mode = downsample_mode

        if self._mid_channels is None:
            feature_dim = round(self._out_channels * self._expand_ratio)
        else:
            feature_dim = self._mid_channels

        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        self._mid_channels = feature_dim

        self.conv1 = Mo.Sequential(OrderedDict([
            ('conv', Mo.Conv(self._in_channels, feature_dim, (1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)),
            ('bn', Mo.BatchNormalization(feature_dim, 4)),
            ('act', build_activation(self._act_func)),
        ]))

        pad = get_same_padding(self._kernel)
        self.conv2 = Mo.Sequential(OrderedDict([
            ('conv', Mo.Conv(feature_dim, feature_dim, kernel, pad=pad, stride=stride, group=group, with_bias=False)),
            ('bn', Mo.BatchNormalization(feature_dim, 4)),
            ('act', build_activation(self._act_func)),
        ]))

        self.conv3 = Mo.Sequential(OrderedDict([
            ('conv', Mo.Conv(feature_dim, self._out_channels, (1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)),
            ('bn', Mo.BatchNormalization(self._out_channels, 4)),
        ]))

        if stride == (1, 1) and in_channels == out_channels:
            self.downsample = IdentityLayer(in_channels, out_channels)
        elif self._downsample_mode == 'conv':
            self.downsample = Mo.Sequential(OrderedDict([
                ('conv', Mo.Conv(in_channels, out_channels, (1, 1), pad=(0, 0), stride=stride, with_bias=False)),
                ('bn', Mo.BatchNormalization(out_channels, 4)),
            ]))
        elif self._downsample_mode == 'avgpool_conv':
            self.downsample = Mo.Sequential(OrderedDict([
                ('avg_pool', Mo.AvgPool(kernel=stride, stride=stride, pad=(0, 0))),
                ('conv', Mo.Conv(in_channels, out_channels, (1, 1), pad=(0, 0), stride=(1, 1), with_bias=False)),
                ('bn', Mo.BatchNormalization(out_channels, 4)),
            ]))
        else:
            raise NotImplementedError

        self.final_act = build_activation(self._act_func)

    def call(self, x):

        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + residual
        x = self.final_act(x)

        return x


class IdentityLayer(My2DLayer):
    def __init__(self, in_channels, out_channels,
                 use_bn=False, use_se=False, act_func=None, dropout=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, use_se, act_func, dropout, ops_order)

        self._scope_name = f'<identitylayer at {hex(id(self))}>'

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return 'Identity'

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)


class LinearLayer(Mo.Module):
    def __init__(self, in_features, out_features, bias=True,
                 use_bn=False, act_func=None, dropout=0, ops_order='weight_bn_act'):

        super(LinearLayer, self).__init__()

        self._scope_name = f'<linearlayer at {hex(id(self))}>'
        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias

        self._use_bn = use_bn
        self._act_func = act_func
        self._dropout = dropout
        self._ops_order = ops_order

        modules = {}
        # batch norm
        if self._use_bn:
            if self.bn_before_weight:
                modules['bn'] = Mo.BatchNormalization(in_features, 2)
            else:
                modules['bn'] = Mo.BatchNormalization(out_features, 2)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self._act_func)
        # dropout
        if self._dropout > 0:
            modules['dropout'] = Mo.Dropout(self._dropout)
        else:
            modules['dropout'] = None
        # linear
        modules['weight'] = {
            'linear': Mo.Linear(self._in_features, self._out_features, bias=self._bias)}

        # add modules
        module_dict = OrderedDict()
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    module_dict['dropout'] = modules['dropout']
                for key in modules['weight']:
                    module_dict[key] = modules['weight'][key]
            else:
                module_dict[op] = modules[op]
        self.linearlayer = Mo.Sequential(module_dict)

    @property
    def ops_list(self):
        return self._ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def call(self, input):
        for module in self.linearlayer:
            input = module(input)
        return input

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)


def build_activation(act_func, inplace=False):
    if act_func == 'relu':
        return Mo.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return Mo.ReLU6()
    elif act_func == 'h_swish':
        return Hswish()
    elif act_func is None or act_func == 'none':
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


def adjust_bn_according_to_idx_ver1(bn, idx):
    bn._beta = F.gather(bn._beta, idx, 1)
    bn._gamma = F.gather(bn._gamma, idx, 1)
    bn._mean = F.gather(bn._mean, idx, 1)
    bn._var = F.gather(bn._var, idx, 1)


def adjust_bn_according_to_idx(bn, idx):
    bn._beta.d = np.stack([bn._beta.d[:, i, :, :] for i in idx], axis=1)
    bn._gamma.d = np.stack([bn._gamma.d[:, i, :, :] for i in idx], axis=1)
    bn._mean.d = np.stack([bn._mean.d[:, i, :, :] for i in idx], axis=1)
    bn._var.d = np.stack([bn._var.d[:, i, :, :] for i in idx], axis=1)
