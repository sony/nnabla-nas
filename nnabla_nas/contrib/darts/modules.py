import nnabla.functions as F

from ... import module as Mo


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
