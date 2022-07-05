from .... import module as Mo
from ..mobilenet.modules import CANDIDATES


class ChoiceBlock(Mo.Module):
    def __init__(self, in_channels, out_channels, stride, ops, rng=None, name=''):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride

        operators = [CANDIDATES[k](in_channels, out_channels, stride, name)
                     for k in ops]

        self._mixed = Mo.MixedOp(operators, mode='fair', name=name, rng=rng)

    def call(self, input):
        return self._mixed(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'stride={self._stride}')
