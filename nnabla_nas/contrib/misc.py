import nnabla.functions as F
import numpy as np
from nnabla import logger
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
        Mo.Module.__init__(self)
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
        Mo.Module.__init__(self)
        self._drop_prob = drop_prob

    def call(self, input):
        if self._drop_prob == 0:
            return input
        mask = F.rand(shape=(input.shape[0], 1, 1, 1))
        mask = F.greater_equal_scalar(mask, self._drop_prob)
        out = F.mul_scalar(input, 1. / (1 - self._drop_prob))
        out = F.mul2(out, mask)
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
            this module has learnable batchnorm parameters. Defaults to `True`.

    """

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None, affine=True):
        Mo.Module.__init__(self)

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

class ConvBNReLU6(Mo.Module):
    r"""Convolution-BatchNormalization-ReLU6 layer.

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
        fix_parameters (bool, optinal): A boolean value that when set to `False`,
            this module has learnable batchnorm parameters. Defaults to `False`.

    """

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None, with_bias=True, group=1, fix_parameters=False):
        Mo.Module.__init__(self)

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._pad = pad
        self._stride = stride
        self._with_bias = with_bias
        self._group = group
        self._fix_parameters = fix_parameters

        self._operators = Mo.Sequential(
            Mo.Conv(in_channels, out_channels, kernel=kernel,
                    stride=stride, pad=pad, group=self._group,
                    with_bias=with_bias),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4,
                                  fix_parameters=fix_parameters),
            Mo.ReLU6()
        )

    def call(self, input):
        return self._operators(input)

    def __extra_repr__(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'pad={self._pad}, '
                f'with_bias={self._with_bias}, '
                f'group={self._group}, '
                f'fix_parameters={self._fix_parameters}')

class InvertedResidualConv(Mo.Module):
    r"""Inverted Residual Convolution as defined in the 
    MobileNetV2 paper [Sandler2018].

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
        expansion_factor (:obj:`int`): Expansion factor
        fix_parameters (bool, optinal): A boolean value that when set to `False`,
            this module has learnable batchnorm parameters. Defaults to `False`.

    """

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None, expansion_factor=6,
                 fix_parameters=False):
        Mo.Module.__init__(self)

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._pad = pad
        self._stride = stride
        self._expansion_factor=expansion_factor
        self._fix_parameters = fix_parameters

        self._operators = Mo.Sequential()

        #number of feature maps in the middle layers
        hmaps = round(in_channels * self._expansion_factor)
        
        #PW1 (only if expansion)
        if self._expansion_factor > 1:
            self._operators.append(ConvBNReLU6(in_channels, hmaps, kernel=(1, 1),
                        stride=(1, 1), pad=(0, 0), with_bias=False))
        
        #DW2
        self._operators.append(ConvBNReLU6(hmaps, hmaps, kernel=self._kernel,
                    stride=self._stride, pad=self._pad, group= hmaps, with_bias=False))
            
        #PW3 
        self._operators.append(ConvBNReLU6(hmaps, self._out_channels, kernel=(1, 1),
                    stride=(1, 1), pad=(0, 0), with_bias=False))
    
    def call(self, input):
        y = self._operators(input)
        if (self._stride == (1, 1) and self._in_channels == self._out_channels):
            y = y + input
        return y


    def __extra_repr__(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'pad={self._pad}, '
                f'expension_factor={self._expansion_factor}, '
                f'fix_parameters={self._fix_parameters}')


class MixedOp(Mo.Module):
    r"""Mixed Operator layer.

    Selects a single operator or a combination of different operators that are
    allowed in this module.

    Args:
        operators (List of `Module`): A list of modules.
        mode (str, optional): The selecting mode for this module. Defaults to
            `sample`. Possible modes are `sample`, `full`, or `max`.
        alpha (Parameter, optional): The weights used to calculate the
            evaluation probabilities. Defaults to None.

    """

    def __init__(self, operators, mode='sample', alpha=None):
        Mo.Module.__init__(self)

        if mode not in ('max', 'sample', 'full'):
            raise ValueError(f'mode={mode} is not supported.')

        self._active = -1  # save the active index
        self._mode = mode
        self._ops = Mo.ModuleList(operators)
        self._alpha = alpha

        if alpha is None:
            n = len(operators)
            alpha_shape = (n,) + (1, 1, 1, 1)
            alpha_init = ConstantInitializer(0.0)
            self._alpha = Mo.Parameter(
                alpha_shape, initializer=alpha_init)

    def call(self, input):
        if self._mode == 'full':
            out = [op(input) for op in self._ops]
            out = F.stack(*out, axis=0)
            out = F.mul2(out, F.softmax(self._alpha, axis=0))
            return F.sum(out, axis=0)

        if self._active < 0:
            logger.warn('The active index was not initialized.')

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
            op.need_grad = (self._active == i)

    def _update_alpha_grad(self):
        """Update the gradients for parameter `alpha`."""
        probs = softmax(self._alpha.d.flat)
        probs[self._active] -= 1
        self._alpha.g = np.reshape(-probs, self._alpha.shape)
        return self

    def __extra_repr__(self):
        return f'num_ops={len(self._ops)}, mode={self._mode}'


class SepConv(Mo.DwConv):
    def __init__(self, out_channels, *args, **kwargs):
        Mo.DwConv.__init__(self, *args, **kwargs)
        self._out_channels = out_channels
        self._conv_module_pw = Mo.Conv(self._in_channels, out_channels, kernel=(1, 1),
                                 pad=None, group=1, rng=self._rng, with_bias=None)

    def call(self, input):
        return self._conv_module_pw(Mo.DwConv.call(self, input))
