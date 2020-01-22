import nnabla.functions as F
from nnabla.initializer import (ConstantInitializer, UniformInitializer,
                                calc_uniform_lim_glorot)

from .batchnorm import BatchNormalization
from .module import Module, Sequential
from .parameter import Parameter
from .relu import ReLU


class Conv(Module):
    """N-D Convolution with a bias term.

    Args:
        in_channels (~nnabla.Variable): Number of convolution kernels (which
            is equal to the number of input channels)
        out_channels (int): Number of convolution kernels (which is equal to
            the number of output channels). For example, to apply convolution
            on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        group (int): Number of groups of channels. This makes connections
            across channels more sparse by grouping connections along map
            direction.
        w_init (:obj:`nnabla.initializer.BaseInitializer`
            or :obj:`numpy.ndarray`):
            Initializer for weight. By default, it is initialized with
            :obj:`nnabla.initializer.UniformInitializer` within the range
            determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.
        b_init (:obj:`nnabla.initializer.BaseInitializer`
            or :obj:`numpy.ndarray`):
            Initializer for bias. By default, it is initialized with zeros if
            `with_bias` is `True`.
        base_axis (int): Dimensions up to `base_axis` are treated as the
            sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will
            not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: N-D array. See
            :obj:`~nnabla.functions.convolution` for the output shape.
    """

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None, dilation=None, group=1,
                 w_init=None, b_init=None,
                 base_axis=1, rng=None, with_bias=True):
        Module.__init__(self)
        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(in_channels, out_channels, tuple(kernel)), rng=rng)

        w_shape = (out_channels, in_channels // group) + tuple(kernel)
        self.W = Parameter(w_shape, initializer=w_init)
        self.b = None

        if with_bias:
            if b_init is None:
                b_init = ConstantInitializer()
            b_shape = (out_channels, )
            self.b = Parameter(b_shape, initializer=b_init)

        self.base_axis = base_axis
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.group = group

    def __call__(self, input):
        return F.convolution(input, self.W, self.b, self.base_axis,
                             self.pad, self.stride, self.dilation, self.group)


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
