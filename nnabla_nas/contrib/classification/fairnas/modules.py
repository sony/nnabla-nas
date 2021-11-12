import numpy as np
from nnabla.initializer import ConstantInitializer
from .... import module as Mo
from ....module import MixedOp as Op
from ..mobilenet.modules import CANDIDATES


class RandIterator(object):
    r"""RandIterator

    Performs permation of the choices and iterate over
    them in a random order. choices are permuted every time each of
    them have been seen exctly once.

    Args:
        size (:obj:`int`): Number of possible choices.
        rng (numpy.random.RandomState): Random generator.
    """
    def __init__(self, size, rng=None):
        self._size = size
        self._rng = rng or np.random.RandomState(123)
        self._reset()

    def _reset(self):
        self._idx = 0
        n = self._size
        self._seq = self._rng.choice(n, p=np.ones(n)/n, replace=False, size=n)

    def next(self):
        if self._idx == self._size:
            self._reset()
        self._idx = self._idx + 1
        return self._seq[self._idx - 1]


class MixedOp(Op):
    r"""Mixed Operator layer.

    Selects a single operator or a combination of different operators that are
    allowed in this module.

    Args:
        operators (List of `Module`): A list of modules.
        alpha (Parameter, optional): The weights used to calculate the
            evaluation probabilities. Defaults to None.
        rng (numpy.random.RandomState): Random generator for random choice.
        name (string): the name of this module
    """
    def __init__(self, operators, alpha=None, rng=None, name=''):
        Op.__init__(self, operators, alpha=alpha, rng=rng, name=name)
        self._active = None  # save the active index
        self._ops = Mo.ModuleList(operators)
        self._alpha = alpha

        if alpha is None:
            n = len(operators)
            shape = (n,) + (1, 1, 1, 1)
            init = ConstantInitializer(0.0)
            self._alpha = Mo.Parameter(shape, initializer=init)

        self._rand = RandIterator(len(operators), rng=rng)

    def call(self, input):
        # update active index
        self._active = self._rand.next()
        for i, op in enumerate(self._ops):
            op.apply(is_active=(self._active == i))
        return self._ops[self._active](input)


class ChoiceBlock(Mo.Module):
    def __init__(self, in_channels, out_channels, stride, ops, rng=None, name=''):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride

        self._mixed = MixedOp(
            operators=[CANDIDATES[k](in_channels, out_channels, stride, name)
                       for k in ops],
            name=name, rng=rng
        )

    def call(self, input):
        return self._mixed(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'stride={self._stride}')
