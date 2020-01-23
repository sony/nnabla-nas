import nnabla.functions as F
from nnabla.initializer import (ConstantInitializer, UniformInitializer,
                                calc_uniform_lim_glorot)

from .module import Module
from .parameter import Parameter


class Conv(Module):
    """N-D Convolution layer.

    Args:
        in_channels (int): Number of convolution kernels (which is equal to
            the number of input channels)
        out_channels (int): Number of convolution kernels (which is equal to
            the number of output channels). For example, to apply convolution
            on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`, optional): Padding sizes for
            dimensions. Defaults to None.
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
        dilation (:obj:`tuple` of :obj:`int`, optional): Dilation sizes for
            dimensions. Defaults to None.
        group (int, optional): Number of groups of channels. This makes
            connections across channels more sparse by grouping connections
            along map direction. Defaults to 1.
        w_init (:obj:`nnabla.initializer.BaseInitializer`
            or :obj:`numpy.ndarray`, optional):
            Initializer for weight. By default, it is initialized with
            :obj:`nnabla.initializer.UniformInitializer` within the range
            determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.
        b_init (:obj:`nnabla.initializer.BaseInitializer`
            or :obj:`numpy.ndarray`, optional):
            Initializer for bias. By default, it is initialized with zeros if
            `with_bias` is `True`.
        base_axis (int, optional): Dimensions up to `base_axis` are treated as
            the sample dimensions. Defaults to 1.
        fix_parameters (bool): When set to `True`, the weights and biases will
            not be updated.
        rng (numpy.random.RandomState, optional): Random generator for
            Initializer.  Defaults to None.
        with_bias (bool, optional): Specify whether to include the bias term.
            Defaults to True.

    """

    def __init__(self, in_channels, out_channels, kernel, pad=None,
                 stride=None, dilation=None, group=1, w_init=None, b_init=None,
                 base_axis=1, rng=None, with_bias=True):
        super().__init__()

        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(
                    in_channels, out_channels, tuple(kernel)),
                rng=rng
            )

        w_shape = (out_channels, in_channels // group) + tuple(kernel)
        self._W = Parameter(w_shape, initializer=w_init)
        self._b = None

        if with_bias:
            if b_init is None:
                b_init = ConstantInitializer()
            b_shape = (out_channels, )
            self._b = Parameter(b_shape, initializer=b_init)

        self._base_axis = base_axis
        self._pad = pad
        self._stride = stride
        self._dilation = dilation
        self._group = group
        self._kernel = kernel
        self._in_channels = in_channels
        self._out_channels = out_channels

    def call(self, input):
        return F.convolution(input, self._W, self._b, self._base_axis,
                             self._pad, self._stride, self._dilation,
                             self._group)

    def __extra_repr__(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'pad={self._pad}, '
                f'dilation={self._dilation}, '
                f'group={self._group}, '
                f'with_bias={self._b is not None}')


Conv1D = Conv
Conv2D = Conv
Conv3D = Conv
ConvNd = Conv
