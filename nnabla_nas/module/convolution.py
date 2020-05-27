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

import nnabla as nn
import nnabla.functions as F
from nnabla.initializer import ConstantInitializer
from nnabla.initializer import UniformInitializer
from nnabla.initializer import calc_uniform_lim_glorot

from .module import Module
from .parameter import Parameter


class Conv(Module):
    r"""N-D Convolution layer.

    Args:
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`, optional): Padding sizes for
            dimensions. Defaults to None.
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
        dilation (:obj:`tuple` of :obj:`int`, optional): Dilation sizes for
            dimensions. Defaults to None.
        group (int, optional): Number of groups of channels. This makes
            connections across channels more sparse by grouping connections
            along map direction. Defaults to 1.
        w_init (:obj:`nnabla.initializer.BaseInitializer`
            or :obj:`numpy.ndarray`, optional):
            Initializer for weight. By default, it is initialized with
            :obj:`nnabla.initializer.UniformInitializer` within the range
            determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.
        b_init (:obj:`nnabla.initializer.BaseInitializer`
            or :obj:`numpy.ndarray`, optional):
            Initializer for bias. By default, it is initialized with zeros if
            `with_bias` is `True`.
        base_axis (:obj:`int`, optional): Dimensions up to `base_axis` are
            treated as the sample dimensions. Defaults to 1.
        fix_parameters (bool, optional): When set to `True`, the weights and
            biases will not be updated. Defaults to `False`.
        rng (numpy.random.RandomState, optional): Random generator for
            Initializer.  Defaults to None.
        with_bias (bool, optional): Specify whether to include the bias term.
            Defaults to `True`.
        channel_last(bool, optional): If True, the last dimension is
            considered as channel dimension, a.k.a NHWC order. Defaults to
            `False`.
    """

    def __init__(self, in_channels, out_channels, kernel, pad=None,
                 stride=None, dilation=None, group=1, w_init=None, b_init=None,
                 base_axis=1, fix_parameters=False, rng=None, with_bias=True,
                 channel_last=False):
        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(
                    in_channels, out_channels, tuple(kernel)),
                rng=rng
            )

        w_shape = (out_channels, in_channels // group) + tuple(kernel)
        b_shape = (out_channels, )

        self._b = None
        if with_bias and b_init is None:
            b_init = ConstantInitializer()

        if fix_parameters:
            self._W = nn.Variable.from_numpy_array(w_init(w_shape))
            if with_bias:
                self._b = nn.Variable.from_numpy_array(b_init(b_shape))
        else:
            self._W = Parameter(w_shape, initializer=w_init)
            if with_bias:
                self._b = Parameter(b_shape, initializer=b_init)

        self._base_axis = base_axis
        self._pad = pad
        self._stride = stride
        self._dilation = dilation
        self._group = group
        self._kernel = kernel
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._channel_last = channel_last
        self._fix_parameters = fix_parameters
        self._rng = rng

    def call(self, input):
        return F.convolution(input, self._W, self._b, self._base_axis,
                             self._pad, self._stride, self._dilation,
                             self._group, self._channel_last)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'pad={self._pad}, '
                f'dilation={self._dilation}, '
                f'base_axis={self._base_axis}, '
                f'group={self._group}, '
                f'with_bias={self._b is not None}, '
                f'fix_parameters={self._fix_parameters}, '
                f'channel_last={self._channel_last}')


class DwConv(Module):
    r"""N-D Depthwise Convolution layer.

    Args:
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`, optional): Padding sizes for
            dimensions. Defaults to None.
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
        dilation (:obj:`tuple` of :obj:`int`, optional): Dilation sizes for
            dimensions. Defaults to None.
        multiplier (:obj:`int`, optional): Number of output feature maps per
            input feature map. Defaults to 1.
        w_init (:obj:`nnabla.initializer.BaseInitializer`
            or :obj:`numpy.ndarray`, optional):
            Initializer for weight. By default, it is initialized with
            :obj:`nnabla.initializer.UniformInitializer` within the range
            determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.
        b_init (:obj:`nnabla.initializer.BaseInitializer`
            or :obj:`numpy.ndarray`, optional):
            Initializer for bias. By default, it is initialized with zeros if
            `with_bias` is `True`.
        base_axis (:obj:`int`, optional): Dimensions up to `base_axis` are
            treated as the sample dimensions. Defaults to 1.
        fix_parameters (bool, optional): When set to `True`, the weights and
            biases will not be updated. Defaults to `False`.
        rng (numpy.random.RandomState, optional): Random generator for
            Initializer.  Defaults to None.
        with_bias (bool, optional): Specify whether to include the bias term.
            Defaults to `True`.

    References:
        - F. Chollet: Chollet, Francois. "Xception: Deep Learning with
        Depthwise Separable Convolutions. https://arxiv.org/abs/1610.02357
    """

    def __init__(self, in_channels, kernel, pad=None, stride=None,
                 dilation=None, multiplier=1, w_init=None, b_init=None,
                 base_axis=1, fix_parameters=False, rng=None, with_bias=True):
        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(
                    in_channels, in_channels, tuple(kernel)
                ),
                rng=rng
            )

        if with_bias and b_init is None:
            b_init = ConstantInitializer()

        w_shape = (in_channels,) + tuple(kernel)
        b_shape = (in_channels, )

        self._b = None

        if fix_parameters:
            self._W = nn.Variable.from_numpy_array(w_init(w_shape))
            if with_bias:
                self._b = nn.Variable.from_numpy_array(b_init(b_shape))
        else:
            self._W = Parameter(w_shape, initializer=w_init)
            if with_bias:
                self._b = Parameter(b_shape, initializer=b_init)
        self._pad = pad
        self._stride = stride
        self._dilation = dilation
        self._in_channels = in_channels
        self._rng = rng
        self._with_bias = with_bias
        self._fix_parameters = fix_parameters
        self._kernel = kernel
        self._base_axis = base_axis
        self._multiplier = multiplier

    def call(self, input):
        return F.depthwise_convolution(input, self._W, self._b,
                                       self._base_axis, self._pad,
                                       self._stride, self._dilation,
                                       self._multiplier)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'pad={self._pad}, '
                f'dilation={self._dilation}, '
                f'multiplier={self._multiplier}, '
                f'base_axis={self._base_axis}, '
                f'with_bias={self._b is not None}, '
                f'fix_parameters={self._fix_parameters}')
