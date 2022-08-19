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

from .... import module as Mo
from .utils.common_tools import get_same_padding, min_divisible_value, make_divisible


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
        XceptionBlock.__name__: XceptionBlock,
        BottleneckResidualBlock.__name__: BottleneckResidualBlock,
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


def get_active_padding(kernel, stride, dilation):
    r"""
        Returns the padding required (as a tuple) such that
        [output_size = input_size/stride] after convolution
        Assumption: padding is equal in both dimensions

    Args:
        kernel (int): kernel size as an int (assuming equal in both dimensions)
        stride (int): stride size as an int (assuming equal in both dimensions)
        dilation (int): dilation size as an int (assuming equal in both dimensions)
    """
    pad = math.ceil(stride * ((kernel + (kernel-1)*(dilation-1))/stride - 1) / 2)
    return (pad, pad)


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


class DWSeparableConv(Mo.Module):

    r"""Depthwise-Separable Conv Layer

    This is an implementation of a depthwise-separable convolution layer.
    Structure followed:
        DepthwiseConv-PointwiseConv-BatchNorm(Optional)-Activation(Optional)

    Args:
        in_channels (int): Number of convolution kernels in the
            depthwise convolution (which is equal to the number
            of input channels).
        out_channels (int): Number of convolution kernels in the pointwise
            convolution (which is equal to the number of output channels).
        kernel (tuple of int, optional): Convolution kernel size for the
            depthwise convolution. Defaults to (1, 1)
        stride (tuple of int, optional): Stride sizes for the depthwise
            convolution. Defaults to (1, 1).
        pad (tuple of int, optional): Padding sizes for the depthwise
            convolution layer. Defaults to (0, 0)
        dilation (tuple of int, optional): Dilation sizes for the
            depthwise convolution layer. Defaults to (1, 1)
        use_bn (bool, optional): If True, BatchNormalization layer is added.
            Defaults to True.
        act_func (str, optional) Type of activation. Defaults to None.
    """

    def __init__(
            self, in_channels, out_channels, kernel=(1, 1),
            stride=(1, 1), pad=(0, 0), dilation=(1, 1),
            use_bn=True, act_fn=None):
        super(DWSeparableConv, self).__init__()

        self.dwconv = Mo.Conv(in_channels, in_channels, kernel, pad=pad, dilation=dilation,
                              stride=stride, with_bias=False, group=in_channels)
        self.pointwise = Mo.Conv(in_channels, out_channels, (1, 1), stride=(1, 1),
                                 pad=(0, 0), dilation=(1, 1), group=1, with_bias=False)

        self.bn = Mo.BatchNormalization(out_channels, 4) if use_bn else None
        self.act = build_activation(act_fn)

    def call(self, x):
        x = self.dwconv(x)
        x = self.pointwise(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x

    @staticmethod
    def build_from_config(config):
        return DWSeparableConv(**config)

    def extra_repr(self):
        return get_extra_repr(self)


class XceptionBlock(Mo.Module):

    r"""XceptionBlock

    This is the primary static XceptionBlock used in
    EntryFlow, MiddleFlow and ExitFlow of the Xception
    Models

    Args:
        in_channels (int): Number of convolution kernels in the input
            layer (which is equal to the number of input channels).
        out_channels (int): Number of convolution kernels in the last
            layer (which is equal to the number of output channels).
        reps (int): Number of ReLU+DWSeparableConv layers in the block.
            If reps==1, grow_first and expand_ratio will be ignored.
        kernel (tuple of int, optional): Convolution kernel size for the
            DWSeparableConv layers in this block. Defaults to (3, 3)
        stride (tuple of int, optional): Stride sizes for residual
            connections and maxpool layers. Defaults to (1, 1).
        start_with_relu (bool, optional): Sets the first layer of the
            block as ReLU, otherwise skips this. Defaults to True
        grow_first (bool, optional): Sets mid_channels to out_channels if True,
            else sets mid_channels to in_channels. The effect of this
            argument is rendered useless when expand_ratio is not None.
            Defaults to True
        expand_ratio (float, optional): Used for calculating the number
            of mid_channels. This is especially useful when this block
            is constructed by DynamicXPLayer for building the subnet.
            Defaults to None
    """

    def __init__(
            self, in_channels, out_channels, reps, kernel=(3, 3),
            stride=(1, 1), start_with_relu=True, grow_first=True,
            expand_ratio=None):
        super(XceptionBlock, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        if out_channels != in_channels or stride != (1, 1):
            self._skip = Mo.Conv(in_channels, out_channels,
                                 (1, 1), stride=stride, with_bias=False)
            self._skipbn = Mo.BatchNormalization(out_channels, 4)
        else:
            self._skip = None

        rep = []
        mid_channels = out_channels if grow_first else in_channels
        if expand_ratio is not None:
            # override mid_channels if expand_ratio is given, since this
            # refers to keeping the current active number of mid_channels
            # as supplied by `get_active_subnet_config` of DynamicXPLayer
            mid_channels = make_divisible(round(in_channels * expand_ratio))

        # calculate padding required in the dwconv of DWSeparableConv
        pad_sep = get_active_padding(kernel[0], 1, 1)

        # if reps==1, we just have a single DWSeparableConv and hence,
        # mid_channels is not required. grow_first and expand_ratio
        # are ignored completely.
        for idx in range(1, reps + 1):
            inp_c = in_channels if idx == 1 else mid_channels
            out_c = out_channels if idx == reps else mid_channels

            rep.append((f'relu{idx}', Mo.ReLU(inplace=True)))
            rep.append((f'sepconv{idx}', DWSeparableConv(inp_c, out_c,
                        kernel=kernel, stride=(1, 1), pad=pad_sep)))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = ('relu1', Mo.ReLU(inplace=False))

        if stride != (1, 1):
            rep.append(('maxpool', Mo.MaxPool((3, 3), stride=stride, pad=(1, 1))))
        self.rep = Mo.Sequential(OrderedDict(rep))

    def call(self, inp):
        x = self.rep(inp)

        if self._skip is not None:
            skip = self._skip(inp)
            skip = self._skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @staticmethod
    def build_from_config(config):
        return XceptionBlock(**config)

    def extra_repr(self):
        return get_extra_repr(self)


class BottleneckResidualBlock(Mo.Module):

    r"""BottleneckResidualBlock

    Args:
        in_channels (int): Number of convolution kernels in the input
            layer (which is equal to the number of input channels).
        out_channels (int): Number of convolution kernels in the last
            layer (which is equal to the number of output channels).
        kernel (tuple of int, optional): Convolution kernel size for the
            Conv layers in this block. Defaults to (3, 3)
        stride (tuple of int, optional): Stride sizes for residual
            connections. Defaults to (1, 1).
        expand_ratio (float, optional): Used for calculating the number
            of mid_channels. Defaults to None.
        mid_channels (int, optional): Number of mid channels of the
           bottleneck. Defaults to None.
        act_func (str, optional): Type of activation. Defaults to 'relu'.
        downsample_mode (str, optional): Downsample method for the
           residual connection. Defaults to 'avgpool_conv'.
    """

    def __init__(self, in_channels, out_channels, kernel=(3, 3),
                 stride=(1, 1), expand_ratio=0.25, mid_channels=None,
                 act_func='relu', downsample_mode='avgpool_conv'):
        super(BottleneckResidualBlock, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        self._kernel = kernel
        self._stride = stride
        self._expand_ratio = expand_ratio
        self._mid_channels = mid_channels
        self._act_func = act_func

        self._downsample_mode = downsample_mode

        if self._mid_channels is None:
            feature_dim = round(self._out_channels * self._expand_ratio)
        else:
            feature_dim = self._mid_channels

        feature_dim = make_divisible(feature_dim)
        self._mid_channels = feature_dim

        self.conv1 = Mo.Sequential(OrderedDict([
            ('conv', Mo.Conv(self._in_channels, feature_dim, (1, 1), stride=(1, 1), pad=(0, 0), with_bias=False)),
            ('bn', Mo.BatchNormalization(feature_dim, 4)),
            ('act', build_activation(self._act_func))
        ]))

        pad = get_same_padding(self._kernel)
        self.conv2 = Mo.Sequential(OrderedDict([
            ('conv', Mo.Conv(feature_dim, feature_dim, kernel, stride=stride, pad=pad, with_bias=False)),
            ('bn', Mo.BatchNormalization(feature_dim, 4)),
            ('act', build_activation(self._act_func))
        ]))

        self.conv3 = Mo.Sequential(OrderedDict([
            ('conv', Mo.Conv(
                feature_dim, self._out_channels, (1, 1), stride=(1, 1), pad=(0, 0), with_bias=False)),
            ('bn', Mo.BatchNormalization(self._out_channels, 4))
        ]))

        if stride == (1, 1) and in_channels == out_channels:
            self.downsample = Mo.Identity()
        elif self._downsample_mode == 'conv':
            self.downsample = Mo.Sequential(OrderedDict([
                ('conv', Mo.Conv(in_channels, out_channels, (1, 1), stride=stride, pad=(0, 0), with_bias=False)),
                ('bn', Mo.BatchNormalization(out_channels, 4)),
            ]))
        elif self._downsample_mode == 'avgpool_conv':
            self.downsample = Mo.Sequential(OrderedDict([
                ('avg_pool', Mo.AvgPool(stride, stride=stride, pad=(0, 0), ignore_border=False)),
                ('conv', Mo.Conv(in_channels, out_channels, (1, 1), stride=(1, 1), pad=(0, 0), with_bias=False)),
                ('bn', Mo.BatchNormalization(out_channels, 4)),
            ]))
        else:
            raise ValueError()

        self.final_act = build_activation(self._act_func)

    def call(self, x):
        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + residual
        x = self.final_act(x)
        return x

    @staticmethod
    def build_from_config(config):
        return BottleneckResidualBlock(**config)

    def extra_repr(self):
        return get_extra_repr(self)
