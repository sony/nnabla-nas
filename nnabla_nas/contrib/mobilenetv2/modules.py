from ... import module as Mo


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
        self._expansion_factor = expansion_factor
        self._fix_parameters = fix_parameters

        self._operators = Mo.Sequential()

        # number of feature maps in the middle layers
        hmaps = round(in_channels * self._expansion_factor)

        # PW1 (only if expansion)
        if self._expansion_factor > 1:
            self._operators.append(ConvBNReLU6(in_channels, hmaps, kernel=(1, 1),
                                               stride=(1, 1), pad=(0, 0), with_bias=False))

        # DW2
        self._operators.append(ConvBNReLU6(hmaps, hmaps, kernel=self._kernel,
                                           stride=self._stride, pad=self._pad, group=hmaps, with_bias=False))

        # PW3
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
