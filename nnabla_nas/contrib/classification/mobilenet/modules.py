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

CANDIDATES = OrderedDict([
    ('MB1 3x3',
        lambda inc, outc, s: InvertedResidual(inc, outc, s,
                                              expand_ratio=1, kernel=(3, 3))),
    ('MB3 3x3',
        lambda inc, outc, s: InvertedResidual(inc, outc, s,
                                              expand_ratio=3, kernel=(3, 3))),
    ('MB6 3x3',
        lambda inc, outc, s: InvertedResidual(inc, outc, s,
                                              expand_ratio=6, kernel=(3, 3))),
    ('MB1 5x5',
        lambda inc, outc, s: InvertedResidual(inc, outc, s,
                                              expand_ratio=1, kernel=(5, 5))),
    ('MB3 5x5',
        lambda inc, outc, s: InvertedResidual(inc, outc, s,
                                              expand_ratio=3, kernel=(5, 5))),
    ('MB6 5x5',
        lambda inc, outc, s: InvertedResidual(inc, outc, s,
                                              expand_ratio=6, kernel=(5, 5))),
    ('MB1 7x7',
        lambda inc, outc, s: InvertedResidual(inc, outc, s,
                                              expand_ratio=1, kernel=(7, 7))),
    ('MB3 7x7',
        lambda inc, outc, s: InvertedResidual(inc, outc, s,
                                              expand_ratio=3, kernel=(7, 7))),
    ('MB6 7x7',
        lambda inc, outc, s: InvertedResidual(inc, outc, s,
                                              expand_ratio=6, kernel=(7, 7))),
    ('skip_connect', lambda inc, outc, s: Mo.Identity())
])


class ConvBNReLU(Mo.Sequential):
    r"""Convolution-BatchNormalization-ReLU layer.

    Args:
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3,5).
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
    """

    def __init__(self, in_channels, out_channels, kernel=(3, 3),
                 stride=(1, 1), group=1):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._stride = stride
        self._pad = ((kernel[0] - 1)//2, (kernel[1] - 1)//2)

        super(ConvBNReLU, self).__init__(
            Mo.Conv(in_channels, out_channels, self._kernel,
                    stride=self._stride, pad=self._pad, group=group,
                    with_bias=False),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4),
            Mo.ReLU6()
        )

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'pad={self._pad}')


class InvertedResidual(Mo.Module):
    """The Inverted-Resisual layer.

    Args:
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3, 5).
        expand_ratio(:obj:`int`): The expand ratio.
    """

    def __init__(self, in_channels, out_channels, stride, kernel=(3, 3),
                 expand_ratio=1):
        assert stride in [1, 2]

        self._stride = stride
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._expand_ratio = expand_ratio

        hidden_dim = int(round(in_channels * expand_ratio))
        self._use_res_connect = (self._stride == 1 and
                                 in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, (1, 1)))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, kernel=kernel,
                       stride=(stride, stride), group=hidden_dim),
            Mo.Conv(hidden_dim, out_channels, kernel=(1, 1), stride=(1, 1),
                    with_bias=False),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4)
        ])

        self._conv = Mo.Sequential(*layers)

    def call(self, x):
        if self._use_res_connect:
            return x + self._conv(x)
        return self._conv(x)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'expand_ratio={self._expand_ratio}')


class ChoiceBlock(Mo.Module):
    def __init__(self, in_channels, out_channels, stride,
                 ops, mode='sample'):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._mode = mode

        self._mixed = Mo.MixedOp(
            operators=[CANDIDATES[k](in_channels, out_channels, stride)
                       for k in ops],
            mode=mode
        )

    def call(self, input):
        return self._mixed(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'stride={self._stride}, '
                f'mode={self._mode}')
