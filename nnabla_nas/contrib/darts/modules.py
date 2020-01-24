import nnabla.functions as F

from ... import module as Mo


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


class DilConv(Module):
    """Dilated depthwise separable convolution.
    """

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None, affine=True):
        super().__init__()
        self._conv = Sequential(
            ReLU(),
            Conv(in_channels=in_channels, out_channels=in_channels,
                 kernel=kernel, pad=pad, stride=stride,
                 dilation=(2, 2), group=in_channels, with_bias=False),
            Conv(in_channels=in_channels, out_channels=out_channels,
                 kernel=(1, 1), with_bias=False),
            BatchNormalization(n_features=out_channels,
                               n_dims=4, fix_parameters=not affine)
        )

    def __call__(self, input):
        return self._conv(input)


class SepConv(Module):
    """Separable convolution."""

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None, affine=True):
        super().__init__()
        self._conv = Sequential(
            ReLU(),
            Conv(in_channels=in_channels, out_channels=in_channels,
                 kernel=kernel, pad=pad, stride=stride,
                 group=in_channels, with_bias=False),
            Conv(in_channels=in_channels, out_channels=in_channels,
                 kernel=(1, 1), with_bias=False),
            BatchNormalization(n_features=in_channels,
                               n_dims=4, fix_parameters=not affine),
            ReLU(),
            Conv(in_channels=in_channels, out_channels=in_channels,
                 kernel=kernel, pad=pad, stride=(1, 1), group=in_channels,
                 with_bias=False),
            Conv(in_channels=in_channels, out_channels=out_channels,
                 kernel=(1, 1), with_bias=False),
            BatchNormalization(n_features=out_channels,
                               n_dims=4, fix_parameters=not affine)
        )

    def __call__(self, input):
        return self._conv(input)


class ChoiceBlock(Mo.Module):
    def __init__(self, in_channels, out_channels,
                 is_reduced=False, mode='full', alpha=None, affine=True):
        super().__init__()
        self._is_reduced = is_reduced
        stride = (2, 2) if is_reduced else (1, 1)
        self._mixed = Mo.MixedOp(
            operators=[
                Mo.DilConv(in_channels, out_channels, (3, 3),
                           pad=(2, 2), stride=stride, affine=affine),
                Mo.DilConv(in_channels, out_channels, (5, 5),
                           pad=(4, 4), stride=stride, affine=affine),
                Mo.SepConv(in_channels, out_channels,
                           (3, 3), pad=(1, 1), stride=stride, affine=affine),
                Mo.SepConv(in_channels, out_channels,
                           (5, 5), pad=(2, 2), stride=stride, affine=affine),
                Mo.MaxPool(kernel=(3, 3), stride=stride, pad=(1, 1)),
                Mo.AvgPool(kernel=(3, 3), stride=stride, pad=(1, 1)),
                Mo.FactorizedReduce(in_channels, out_channels, affine=affine)
                if is_reduced else Mo.Identity(),
                Mo.Zero(stride)
            ],
            mode=mode,
            alpha=alpha
        )

    def __call__(self, input):
        return self._mixed(input)


class StemConv(Mo.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._conv = Mo.Sequential(
            Mo.Conv(in_channels, out_channels,
                    kernel=(3, 3), pad=(1, 1), with_bias=False),
            Mo.BatchNormalization(out_channels, 4)
        )

    def __call__(self, input):
        return self._conv(input)


class Cell(Mo.Module):
    """Cell in DARTS.
    """

    def __init__(self, num_choices, multiplier, channels, reductions,
                 mode='full', alpha=None, affine=False):
        super().__init__()
        self._multiplier = multiplier
        self._num_choices = num_choices
        # preprocess the inputs
        self._prep = Mo.ModuleList()
        if reductions[0]:
            self._prep.add_module(
                Mo.FactorizedReduce(channels[0], channels[2], affine=affine))
        else:
            self._prep.add_module(
                Mo.ReLUConvBN(channels[0], channels[2], kernel=(1, 1), affine=affine))
        self._prep.add_module(Mo.ReLUConvBN(
            channels[1], channels[2], kernel=(1, 1), affine=affine))
        # build choice blocks
        self._blocks = Mo.ModuleList()
        for i in range(num_choices):
            for j in range(i + 2):
                self._blocks.add_module(
                    ChoiceBlock(in_channels=channels[2],
                                out_channels=channels[2],
                                is_reduced=j < 2 and reductions[1],
                                mode=mode,
                                alpha=alpha[len(self._blocks)],
                                affine=affine)
                )

    def __call__(self, *input):
        """Each cell has two inputs and one output."""
        out = [op(x) for op, x in zip(self._prep, input)]
        offset = 0
        for _ in range(self._num_choices):
            s = sum(self._blocks[offset + j](h) for j, h in enumerate(out))
            offset += len(out)
            out.append(s)
        return F.concatenate(*out[-self._multiplier:], axis=1)
