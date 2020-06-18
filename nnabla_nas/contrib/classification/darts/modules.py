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

import nnabla.functions as F

from .... import module as Mo

CANDIDATES = OrderedDict([
    ('dil_conv_3x3', lambda c, s: DDSConv(c, c, (3, 3), (2, 2), (s, s))),
    ('dil_conv_5x5', lambda c, s: DDSConv(c, c, (5, 5), (4, 4), (s, s))),
    ('sep_conv_3x3', lambda c, s: SepConv(c, c, (3, 3), (1, 1), (s, s))),
    ('sep_conv_5x5', lambda c, s: SepConv(c, c, (5, 5), (2, 2), (s, s))),
    ('max_pool_3x3', lambda c, s: Mo.MaxPool((3, 3), (s, s), (1, 1))),
    ('avg_pool_3x3', lambda c, s: Mo.AvgPool((3, 3), (s, s), (1, 1))),
    ('skip_connect', lambda c, s: FactorizedReduce(c, c) if s > 1
     else Mo.Identity()),
    ('none', lambda c, s: Mo.Zero((s, s)))
])


class DropPath(Mo.Module):
    r"""Drop Path layer.

    Args:
        drop_prob (:obj:`int`, optional): The probability of droping path.
            Defaults to 0.2.
    """

    def __init__(self, drop_prob=0.2):
        self._drop_prob = drop_prob

    def call(self, input):
        if self._drop_prob == 0:
            return input
        mask = F.rand(shape=(input.shape[0], 1, 1, 1))
        mask = F.greater_equal_scalar(mask, self._drop_prob)
        out = F.mul_scalar(input, 1. / (1 - self._drop_prob))
        out = F.mul2(out, mask)
        return out

    def extra_repr(self):
        return f'drop_prob={self._drop_prob}'


class ReLUConvBN(Mo.Module):
    r"""ReLU-Convolution-BatchNormalization layer.

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
    """

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._pad = pad
        self._stride = stride

        self._operators = Mo.Sequential(
            Mo.ReLU(),
            Mo.Conv(in_channels, out_channels, kernel=kernel,
                    stride=stride, pad=pad, with_bias=False),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4)
        )

    def call(self, input):
        return self._operators(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'pad={self._pad}')


class FactorizedReduce(Mo.Module):
    r"""Factorize-Reduction layer.

    Args:
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
    """

    def __init__(self, in_channels, out_channels):
        if out_channels % 2:
            raise ValueError(f'{out_channels} must be even.')

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._relu = Mo.ReLU()
        self._conv_1 = Mo.Conv(in_channels, out_channels // 2, kernel=(1, 1),
                               stride=(2, 2), with_bias=False)
        self._conv_2 = Mo.Conv(in_channels, out_channels // 2, kernel=(1, 1),
                               stride=(2, 2), with_bias=False)
        self._conc = Mo.Merging(mode='concat', axis=1)
        self._bn = Mo.BatchNormalization(n_features=out_channels, n_dims=4)

    def call(self, input):
        out = self._relu(input)
        out = self._conc(
            self._conv_1(out),
            self._conv_2(out[:, :, 1:, 1:])
        )
        out = self._bn(out)
        return out

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}')


class DDSConv(Mo.Module):
    """Dilated depthwise separable convolution layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel (:obj:`tuple` of :obj:`int`): The kernel size.
        pad (:obj:`tuple` of :obj:`int`): Border padding values for each
            spatial axis. Padding will be added both sides of the dimension.
            [default=``(0,) * len(kernel)``].
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
            Defaults to None.
    """

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._pad = pad
        self._stride = stride

        self._conv = Mo.Sequential(
            Mo.ReLU(),
            Mo.Conv(in_channels=in_channels, out_channels=in_channels,
                    kernel=kernel, pad=pad, stride=stride,
                    dilation=(2, 2), group=in_channels, with_bias=False),
            Mo.Conv(in_channels=in_channels, out_channels=out_channels,
                    kernel=(1, 1), with_bias=False),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4)
        )

    def call(self, input):
        return self._conv(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'pad={self._pad}, '
                f'stride={self._stride}')


class SepConv(Mo.Module):
    """Separable convolution.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel (:obj:`tuple` of :obj:`int`): The kernel size.
        pad (:obj:`tuple` of :obj:`int`): Border padding values for each
            spatial axis. Padding will be added both sides of the dimension.
            [default=``(0,) * len(kernel)``].
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
            Defaults to None.
    """

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._pad = pad
        self._stride = stride

        self._conv = Mo.Sequential(
            Mo.ReLU(),
            Mo.Conv(in_channels=in_channels, out_channels=in_channels,
                    kernel=kernel, pad=pad, stride=stride,
                    group=in_channels, with_bias=False),
            Mo.Conv(in_channels=in_channels, out_channels=in_channels,
                    kernel=(1, 1), with_bias=False),
            Mo.BatchNormalization(n_features=in_channels, n_dims=4),
            Mo.ReLU(),
            Mo.Conv(in_channels=in_channels, out_channels=in_channels,
                    kernel=kernel, pad=pad, stride=(1, 1), group=in_channels,
                    with_bias=False),
            Mo.Conv(in_channels=in_channels, out_channels=out_channels,
                    kernel=(1, 1), with_bias=False),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4)
        )

    def call(self, input):
        return self._conv(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'pad={self._pad}, '
                f'stride={self._stride}')


class ChoiceBlock(Mo.Module):
    r"""Choice block layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        is_reduced (bool, optional): Whether it's a reduced block.
        mode (str, optional): The selection mode ('max', 'full', 'sample').
            Defaults to 'full'.
        alpha (Parameter, optional): The parameter for MixedOp. Defaults
            to None.
    """

    def __init__(self, in_channels, out_channels, is_reduced=False,
                 mode='full', alpha=None):
        self._is_reduced = is_reduced
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._mode = mode
        stride = 2 if is_reduced else 1
        self._mixed = Mo.MixedOp(
            operators=[func(in_channels, stride)
                       for func in CANDIDATES.values()],
            mode=mode,
            alpha=alpha
        )

    def call(self, input):
        return self._mixed(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'is_reduced={self._is_reduced}, '
                f'mode={self._mode}')


class StemConv(Mo.Module):
    r"""Stem convolution layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._conv = Mo.Sequential(
            Mo.Conv(in_channels, out_channels,
                    kernel=(3, 3), pad=(1, 1), with_bias=False),
            Mo.BatchNormalization(out_channels, 4)
        )

    def call(self, input):
        return self._conv(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}')


class Cell(Mo.Module):
    r"""Cell layer.

    Args:
        num_choices (int): Number of choices in the cell.
        multiplier (int): Number of multiplier.
        channels (tuple of int): A tuple of channels over previous cells.
        reductions (tuple of bool): A tuple of is_reduced over previous cells.
        mode (str, optional): The selection mode. Defaults to 'full'.
        alpha (Parameter, optional): The parameter for MixedOp. Defaults
            to None.
    """

    def __init__(self, num_choices, multiplier, channels, reductions,
                 mode='full', alpha=None):
        self._multiplier = multiplier
        self._num_choices = num_choices
        self._channels = channels
        self._mode = mode

        # preprocess the inputs
        self._prep = Mo.ModuleList()
        if reductions[0]:
            self._prep.append(
                FactorizedReduce(channels[0], channels[2]))
        else:
            self._prep.append(
                ReLUConvBN(channels[0], channels[2], kernel=(1, 1)))
        self._prep.append(
            ReLUConvBN(channels[1], channels[2], kernel=(1, 1)))

        # build choice blocks
        self._blocks = Mo.ModuleList()
        for i in range(num_choices):
            for j in range(i + 2):
                self._blocks.append(
                    ChoiceBlock(
                        in_channels=channels[2],
                        out_channels=channels[2],
                        is_reduced=j < 2 and reductions[1],
                        mode=mode,
                        alpha=None if alpha is None
                        else alpha[len(self._blocks)]
                    )
                )
        self._concat = Mo.Merging('concat', axis=1)

    def call(self, *input):
        """Each cell has two inputs and one output."""
        out = [op(x) for op, x in zip(self._prep, input)]
        offset = 0
        for _ in range(self._num_choices):
            s = sum(self._blocks[offset + j](h) for j, h in enumerate(out))
            offset += len(out)
            out.append(s)
        out = self._concat(*out[-self._multiplier:])
        return out

    def extra_repr(self):
        return (f'num_choices={self._num_choices}, '
                f'multiplier={self._multiplier}, '
                f'channels={self._channels}, '
                f'mode={self._mode}')
