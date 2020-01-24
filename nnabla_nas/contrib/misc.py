

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.initializer import ConstantInitializer
from scipy.special import softmax

from .. import module as Mo
from .. import utils as ut


class AuxiliaryHeadCIFAR(Mo.Module):
    r"""Auxiliary head used for CIFAR10 dataset.

    Args:
        channels (:obj:`int`): The number of input channels.
        num_classes (:obj:`int`): The number of classes.

    """

    def __init__(self, channels, num_classes):
        super().__init__()
        self._channels = channels
        self._num_classes = num_classes
        self._feature = Mo.Sequential(
            Mo.ReLU(),
            Mo.AvgPool(kernel=(5, 5), stride=(3, 3)),
            Mo.Conv(in_channels=channels, out_channels=128,
                    kernel=(1, 1), with_bias=False),
            Mo.BatchNormalization(n_features=128, n_dims=4),
            Mo.ReLU(),
            Mo.Conv(in_channels=128, out_channels=768,
                    kernel=(2, 2), with_bias=False),
            Mo.BatchNormalization(n_features=768, n_dims=4),
            Mo.ReLU()
        )
        self._classifier = Mo.Linear(in_features=768, out_features=num_classes)

    def call(self, input):
        out = self._feature(input)
        out = self._classifier(out)
        return out

    def __extra_repr__(self):
        return f'channels={self._channels}, num_classes={self._num_classes}'


class DropPath(Mo.Module):
    r"""Drop Path layer.

    Args:
        drop_prob (:obj:`int`, optional): The probability of droping path.
            Defaults to 0.2.

    """

    def __init__(self, drop_prob=0.2):
        super().__init__()
        self._drop_prob = drop_prob

    def call(self, input):
        mask = F.rand(shape=(input.shape[0], 1, 1, 1))
        mask = F.greater_equal_scalar(mask, self.drop_prob)
        out = F.mul_scalar(input, 1. / (1 - self._drop_prob))
        out = F.mul2(out, self.mask)
        return out

    def __extra_repr__(self):
        return f'drop_prob={self._drop_prob}'


class ReLUConvBN(Mo.Module):
    r"""ReLU-Convolution-BatchNormalization layer.

    Args:
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`, optional): Padding sizes for
            dimensions. Defaults to None.
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
        affine (bool, optinal): A boolean value that when set to `True`,
            this module has learnable parameters. Defaults to `True`.
    """

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None, affine=True):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._pad = pad
        self._stride = stride
        self._affine = affine

        self._operators = Mo.Sequential(
            Mo.ReLU(),
            Mo.Conv(in_channels, out_channels, kernel=kernel,
                    stride=stride, pad=pad, with_bias=False),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4,
                                  fix_parameters=not affine)
        )

    def call(self, input):
        return self._operators(input)

    def __extra_repr__(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'pad={self._pad}, '
                f'affine={self._affine}')


class FactorizedReduce(Mo.Module):
    r"""Factorize layer.
    """

    def __init__(self, in_channels, out_channels, affine=True):
        super().__init__()
        if out_channels % 2:
            raise ValueError(f'{out_channels} must be even.')
        self._relu = Mo.ReLU()
        self._conv_1 = Mo.Conv(in_channels, out_channels // 2, kernel=(1, 1),
                               stride=(2, 2), with_bias=False)
        self._conv_2 = Mo.Conv(in_channels, out_channels // 2, kernel=(1, 1),
                               stride=(2, 2), with_bias=False)
        self._conc = Mo.Merging(mode='concat', axis=1)
        self._bn = Mo.BatchNormalization(n_features=out_channels, n_dims=4,
                                         fix_parameters=not affine)

    def call(self, input):
        out = self._relu(input)
        out = self._conc(
            self._conv_1(out),
            self._conv_2(out[:, :, 1:, 1:])
        )
        out = self._bn(out)
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
