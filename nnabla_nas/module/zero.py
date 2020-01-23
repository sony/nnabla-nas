import nnabla.functions as F

from .module import Module


class Zero(Module):
    r"""Zero layer.
    A placeholder zero operator that is argument-insensitive.

    Args:
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to (1, 1).

    """

    def __init__(self, stride=(1, 1)):
        super().__init__()
        self._stride = stride

    def call(self, input):
        if self._stride[0] > 1:
            out = input[:, :, ::self._stride[0], ::self._stride[1]]
            out = F.mul_scalar(out, 0.0)
        else:
            out = F.mul_scalar(input, 0.0)
        return out
