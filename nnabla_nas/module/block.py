import nnabla.functions as F

from .batchnorm import BatchNormalization
from .convolution import Conv
from .module import Module, Sequential
from .relu import ReLU


class Identity(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return input


class Zero(Module):
    def __init__(self, stride=(1, 1)):
        super().__init__()
        self._stride = stride

    def __call__(self, input):
        out = input[:, :, ::self._stride[0], ::self._stride[1]]
        out = F.mul_scalar(out, 0)
        return out


class FactorizedReduce(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 2 == 0
        self._relu = ReLU()
        self._conv_1 = Conv(in_channels, out_channels // 2,
                            kernel=(1, 1), stride=(2, 2), with_bias=False)
        self._conv_2 = Conv(in_channels, out_channels // 2,
                            kernel=(1, 1), stride=(2, 2), with_bias=False)
        self._bn = BatchNormalization(out_channels, 4)

    def __call__(self, input):
        out = self._relu(input)
        out = F.concatenate(
            self._conv_1(out),
            self._conv_2(out[:, :, 1:, 1:]),
            axis=1
        )
        out = self._bn(out)
        return out


class ReLUConvBN(Module):

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None):
        super().__init__()
        self._ops = Sequential(
            ReLU(),
            Conv(in_channels, out_channels, kernel=kernel, stride=stride,
                 pad=pad, with_bias=False),
            BatchNormalization(n_features=out_channels, n_dims=4)
        )

    def __call__(self, x):
        return self._ops(x)
