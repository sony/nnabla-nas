

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.initializer import ConstantInitializer
from scipy.special import softmax

from .. import utils as ut
from .container import ModuleList
from .module import Module
from .parameter import Parameter


class DropPath(Module):
    def __init__(self, drop_prob, batch_size):
        super().__init__()
        self._batch_size = batch_size
        self.keep_prob = 1.0 - drop_prob
        rand = F.rand(shape=(batch_size, 1, 1, 1))
        self.mask = F.greater_equal_scalar(rand, drop_prob)

    def __call__(self, input):
        out = F.mul_scalar(input, 1./self.keep_prob)
        out = F.mul2(out, self.mask)
        return out


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


class MixedOp(Module):
    def __init__(self, operators, mode='sample', alpha=None):
        super().__init__()
        n = len(operators)
        alpha_shape = (n,) + (1, 1, 1, 1)
        alpha_init = ConstantInitializer(0.0)

        self._mode = mode

        self._ops = ModuleList(operators)
        self._alpha = alpha or Parameter(alpha_shape, initializer=alpha_init)
        self._binary = nn.Variable.from_numpy_array(np.zeros(alpha_shape))

        self._active = 0  # save the active index
        self._state = None  # save the states of intermediate outputs

    def __call__(self, input):
        if self._mode == 'full':
            out = F.mul2(self._ops(input), F.softmax(self._alpha, axis=0))
            return F.sum(out, axis=0)
        return self._ops[self._active](input)

    def _update_active_idx(self):
        """Update index of the active operation."""
        # recompute active_idx
        probs = softmax(self._alpha.d.flat)
        self._active = ut.sample(
            pvals=probs,
            mode=self._mode
        )
        for i, op in enumerate(self._ops):
            op.update_grad(self._active == i)

    def _update_alpha_grad(self):
        probs = softmax(self._alpha.d.flat)
        probs[self._active] -= 1
        self._alpha.g = np.reshape(-probs, self._alpha.shape)
        return self


class ReLUConvBN(Module):

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None, affine=True):
        super().__init__()
        self._ops = Sequential(
            ReLU(),
            Conv(in_channels, out_channels, kernel=kernel, stride=stride,
                 pad=pad, with_bias=False),
            BatchNormalization(n_features=out_channels,
                               n_dims=4, fix_parameters=not affine)
        )

    def __call__(self, x):
        return self._ops(x)


class FactorizedReduce(Module):
    def __init__(self, in_channels, out_channels, affine=True):
        super().__init__()
        assert out_channels % 2 == 0
        self._relu = ReLU()
        self._conv_1 = Conv(in_channels, out_channels // 2,
                            kernel=(1, 1), stride=(2, 2), with_bias=False)
        self._conv_2 = Conv(in_channels, out_channels // 2,
                            kernel=(1, 1), stride=(2, 2), with_bias=False)
        self._bn = BatchNormalization(
            out_channels, 4, fix_parameters=not affine)

    def __call__(self, input):
        out = self._relu(input)
        out = F.concatenate(
            self._conv_1(out),
            self._conv_2(out[:, :, 1:, 1:]),
            axis=1
        )
        out = self._bn(out)
        return out
