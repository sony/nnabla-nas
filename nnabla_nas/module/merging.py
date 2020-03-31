import nnabla.functions as F

from .module import Module


class Merging(Module):
    r"""Merging layer.

    Merges a list of NNabla Variables.

    Args:
        mode (str): The merging mode ('concat', 'add', 'mul'), where `concat` indicates that the
            inputs will be concatenated, `add` means the element-wise addition, and `mul` means
            the element-wise multiplication.
        axis (int, optional): The axis for merging when 'concat' is used.
            Defaults to 1.
    """

    def __init__(self, mode, axis=1):
        if mode not in ('concat', 'add', 'mul'):
            raise KeyError(f'{mode} is not supported.')
        self._mode = mode
        self._axis = axis

    def call(self, *input):
        if self._mode == 'concat' and len(input) > 1:
            return F.concatenate(*input, axis=self._axis)
        if self._mode == 'add':
            return sum(input)
        out = input[0]
        for i in range(1, len(input)):
            out = F.mul2(out, input[i])
        return out

    def extra_repr(self):
        return f'mode={self._mode}, axis={self._axis}'
