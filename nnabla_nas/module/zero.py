import nnabla.functions as F

from .module import Module


class Zero(Module):
    r"""Zero layer.
    A placeholder zero operator that is argument-insensitive.

    Args:
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to (1, 1).
    """

    def __init__(self, stride=(1, 1), *args, **kwargs):
        self._stride = stride

    def call(self, input):
        if self._stride[0] > 1:
            input = F.max_pooling(input, kernel=(1, 1), stride=self._stride)
        return F.mul_scalar(input, 0.0)

    def extra_repr(self):
        return f'stride={self._stride}'
