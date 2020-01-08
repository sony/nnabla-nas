import nnabla.functions as F

from .module import Module


class MaxPool(Module):
    def __init__(self, kernel, stride=None, pad=None):
        super().__init__()
        self._kernel = kernel
        self._stride = stride
        self._pad = pad

    def __call__(self, input):
        out = F.max_pooling(input, kernel=self._kernel,
                            stride=self._stride, pad=self._pad)
        return out


class AvgPool(Module):
    def __init__(self, kernel, stride=None, pad=None):
        super().__init__()
        self._kernel = kernel
        self._stride = stride
        self._pad = pad

    def __call__(self, input):
        out = F.average_pooling(input, kernel=self._kernel,
                                stride=self._stride, pad=self._pad)
        return out
