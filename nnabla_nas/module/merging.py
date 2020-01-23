import nnabla.functions as F

from .module import Module


class Merging(Module):
    r"""Merging layer.
    Merging a list of nnabla variables.

    Args:
        mode (str): The merging mode ('concat', 'add').
        axis (int, optional): The axis for merging. Defaults to 1.
    """

    def __init__(self, mode, axis=1):
        super().__init__()
        assert mode != 'add' or (mode == 'add' and axis == 0)
        if mode not in ('concat', 'add'):
            raise KeyError(f'{mode} is not supported.')

        self._mode = mode
        self._axis = axis

    def call(self, *input):
        if self._mode == 'concat':
            out = F.concatenate(*input, axis=self._axis)
        if self._mode == 'add':
            out = F.sum(F.stack(*input, axis=self._axis), axis=self._axis)
        return out

    def __extra_repr__(self):
        return f'mode={self._mode}, axis={self._axis}'
