import nnabla.functions as F

from ... import module as Mo
from ..misc import ReLUConvBN, MixedOp
from collections import OrderedDict

CANDIDATE_FUNC = OrderedDict([
    ('dil_conv_3x3', lambda channels, stride, affine:
        DilConv(channels, channels, (3, 3),
                pad=(2, 2), stride=(stride, stride), affine=affine)),
    ('dil_conv_5x5', lambda channels, stride, affine:
        DilConv(channels, channels, (5, 5), pad=(4, 4),
                stride=(stride, stride), affine=affine)),
    ('sep_conv_3x3', lambda channels, stride, affine:
        SepConv(channels, channels, (3, 3), pad=(1, 1),
                stride=(stride, stride), affine=affine)),
    ('sep_conv_5x5', lambda channels, stride, affine:
        SepConv(channels, channels, (5, 5), pad=(2, 2),
                stride=(stride, stride), affine=affine)),
    ('max_pool_3x3', lambda channels, stride, affine:
        Mo.MaxPool(kernel=(3, 3), stride=(stride, stride), pad=(1, 1))),
    ('avg_pool_3x3', lambda channels, stride, affine:
        Mo.AvgPool(kernel=(3, 3), stride=(stride, stride), pad=(1, 1))),
    ('skip_connect', lambda channels, stride, affine:
        FactorizedReduce(channels, channels,
                         affine=affine) if stride > 1 else Mo.Identity()),
    ('none', lambda channels, stride, affine: Mo.Zero((stride, stride)))
])


class FactorizedReduce(Mo.Module):
    r"""Factorize-Reduction layer.

    Args:
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        affine (bool, optinal): A boolean value that when set to `True`,
            this module has learnable batchnorm parameters. Defaults to `True`.

    """

    def __init__(self, in_channels, out_channels, affine=True):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._affine = affine

        if out_channels % 2:
            raise ValueError(f'{out_channels} must be even.')
        self._relu = Mo.ReLU()
        self._conv_1 = Mo.Conv(in_channels, out_channels // 2, kernel=(1, 1),
                               stride=(2, 2), with_bias=False)
        self._conv_2 = Mo.Conv(in_channels, out_channels // 2, kernel=(1, 1),
                               stride=(2, 2), with_bias=False)
        self._conc = Mo.Merging(mode='concat', axis=1)
        self._bn = Mo.BatchNormalization(n_features=out_channels, n_dims=4,
                                         fix_parameters=not affine)

    def call(self, input):
        out = self._relu(input)
        out = self._conc(
            self._conv_1(out),
            self._conv_2(out[:, :, 1:, 1:])
        )
        out = self._bn(out)
        return out

    def __extra_repr__(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'affine={self._affine}')


class DilConv(Mo.Module):
    """Dilated depthwise separable convolution.
    """

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None, affine=True):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._pad = pad
        self._stride = stride
        self._affine = affine

        self._conv = Mo.Sequential(
            Mo.ReLU(),
            Mo.Conv(in_channels=in_channels, out_channels=in_channels,
                    kernel=kernel, pad=pad, stride=stride,
                    dilation=(2, 2), group=in_channels, with_bias=False),
            Mo.Conv(in_channels=in_channels, out_channels=out_channels,
                    kernel=(1, 1), with_bias=False),
            Mo.BatchNormalization(n_features=out_channels,
                                  n_dims=4, fix_parameters=not affine)
        )

    def call(self, input):
        return self._conv(input)

    def __extra_repr__(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'pad={self._pad}, '
                f'stride={self._stride}, '
                f'affine={self._affine}')


class SepConv(Mo.Module):
    """Separable convolution."""

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None, affine=True):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._pad = pad
        self._stride = stride
        self._affine = affine

        self._conv = Mo.Sequential(
            Mo.ReLU(),
            Mo.Conv(in_channels=in_channels, out_channels=in_channels,
                    kernel=kernel, pad=pad, stride=stride,
                    group=in_channels, with_bias=False),
            Mo.Conv(in_channels=in_channels, out_channels=in_channels,
                    kernel=(1, 1), with_bias=False),
            Mo.BatchNormalization(n_features=in_channels,
                                  n_dims=4, fix_parameters=not affine),
            Mo.ReLU(),
            Mo.Conv(in_channels=in_channels, out_channels=in_channels,
                    kernel=kernel, pad=pad, stride=(1, 1), group=in_channels,
                    with_bias=False),
            Mo.Conv(in_channels=in_channels, out_channels=out_channels,
                    kernel=(1, 1), with_bias=False),
            Mo.BatchNormalization(n_features=out_channels,
                                  n_dims=4, fix_parameters=not affine)
        )

    def call(self, input):
        return self._conv(input)

    def __extra_repr__(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'pad={self._pad}, '
                f'stride={self._stride}, '
                f'affine={self._affine}')


class ChoiceBlock(Mo.Module):
    def __init__(self, in_channels, out_channels, is_reduced=False,
                 mode='full', alpha=None, affine=True):
        super().__init__()
        self._is_reduced = is_reduced
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._mode = mode
        self._affine = affine

        stride = 2 if is_reduced else 1
        self._mixed = MixedOp(
            operators=[func(in_channels, stride, affine)
                       for func in CANDIDATE_FUNC.values()],
            mode=mode,
            alpha=alpha
        )

    def call(self, input):
        return self._mixed(input)

    def __extra_repr__(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'is_reduced={self._is_reduced}, '
                f'mode={self._mode}, '
                f'affine={self._affine}')


class StemConv(Mo.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self._conv = Mo.Sequential(
            Mo.Conv(in_channels, out_channels,
                    kernel=(3, 3), pad=(1, 1), with_bias=False),
            Mo.BatchNormalization(out_channels, 4)
        )

    def call(self, input):
        return self._conv(input)

    def __extra_repr__(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}')


class Cell(Mo.Module):
    """Cell in DARTS.
    """

    def __init__(self, num_choices, multiplier, channels, reductions,
                 mode='full', alpha=None, affine=False):
        super().__init__()
        self._multiplier = multiplier
        self._num_choices = num_choices
        self._channels = channels
        self._mode = mode
        self._affine = affine

        # preprocess the inputs
        self._prep = Mo.ModuleList()
        if reductions[0]:
            self._prep.append(
                FactorizedReduce(channels[0], channels[2], affine=affine))
        else:
            self._prep.append(
                ReLUConvBN(channels[0], channels[2], kernel=(1, 1),
                           affine=affine))
        self._prep.append(
            ReLUConvBN(channels[1], channels[2], kernel=(1, 1), affine=affine))

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
                        alpha=alpha[len(self._blocks)],
                        affine=affine
                    )
                )

    def call(self, *input):
        """Each cell has two inputs and one output."""
        out = [op(x) for op, x in zip(self._prep, input)]
        offset = 0
        for _ in range(self._num_choices):
            s = sum(self._blocks[offset + j](h) for j, h in enumerate(out))
            offset += len(out)
            out.append(s)
        return F.concatenate(*out[-self._multiplier:], axis=1)

    def __extra_repr__(self):
        return (f'num_choices={self._num_choices}, '
                f'multiplier={self._multiplier}, '
                f'channels={self._channels}, '
                f'mode={self._mode}, '
                f'affine={self._affine}')
