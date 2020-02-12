from ... import module as Mo
from ..darts.modules import MixedOp


class ConvBNReLU(Mo.Sequential):
    r"""Convolution-BatchNormalization-ReLU layer.

    Args:
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3,5).
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
    """

    def __init__(self, in_channels, out_channels, kernel=(3, 3),
                 stride=(1, 1), group=1):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._stride = stride
        self._pad = ((kernel[0] - 1)//2, (kernel[1] - 1)//2)

        super(ConvBNReLU, self).__init__(
            Mo.Conv(in_channels, out_channels, self._kernel,
                    stride=self._stride, pad=self._pad, group=group,
                    with_bias=False),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4),
            Mo.ReLU6()
        )

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'pad={self._pad}')


class InvertedResidual(Mo.Module):
    """The Inverted-Resisual layer.

    Args:
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3, 5).
        expand_ratio(:obj:`int`): The expand ratio.
    """

    def __init__(self, in_channels, out_channels, stride, kernel=(3, 3),
                 expand_ratio=1):
        assert stride in [1, 2]

        self._stride = stride
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._expand_ratio = expand_ratio

        hidden_dim = int(round(in_channels * expand_ratio))
        self._use_res_connect = (self._stride == 1 and
                                 in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, (1, 1)))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, kernel=kernel,
                       stride=(stride, stride), group=hidden_dim),
            Mo.Conv(hidden_dim, out_channels, kernel=(1, 1), stride=(1, 1),
                    with_bias=False),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4)
        ])

        self._conv = Mo.Sequential(*layers)

    def call(self, x):
        if self._use_res_connect:
            return x + self._conv(x)
        return self._conv(x)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'expand_ratio={self._expand_ratio}')


class ChoiceBlock(Mo.Module):
    def __init__(self, in_channels, out_channels, stride,
                 mode='full', is_skipped=False):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._mode = mode
        self._is_skipped = is_skipped

        ops = [
            InvertedResidual(in_channels, out_channels, stride,
                             expand_ratio=3, kernel=(3, 3)),
            InvertedResidual(in_channels, out_channels, stride,
                             expand_ratio=6, kernel=(3, 3)),
            InvertedResidual(in_channels, out_channels, stride,
                             expand_ratio=3, kernel=(5, 5)),
            InvertedResidual(in_channels, out_channels, stride,
                             expand_ratio=6, kernel=(5, 5)),
            InvertedResidual(in_channels, out_channels, stride,
                             expand_ratio=3, kernel=(7, 7)),
            InvertedResidual(in_channels, out_channels, stride,
                             expand_ratio=6, kernel=(7, 7))
        ]
        ops += [Mo.Identity()] if is_skipped else []
        self._mixed = MixedOp(operators=ops, mode=mode)

    def call(self, input):
        return self._mixed(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'stride={self._stride}, '
                f'mode={self._mode}, '
                f'is_skipped={self._is_skipped}')


# class Conv1x1BN(Mo.Module):
#     r"""Convolution1x1-BatchNormalization layer.

#     Args:
#         in_channels (:obj:`int`): Number of convolution kernels (which is
#             equal to the number of input channels).
#         out_channels (:obj:`int`): Number of convolution kernels (which is
#             equal to the number of output channels). For example, to apply
#             convolution on an input with 16 types of filters, specify 16.
#     """

#     def __init__(self, in_channels, out_channels):
#         self._in_channels = in_channels
#         self._out_channels = out_channels

#         self._operators = Mo.Sequential(
#             Mo.Conv(in_channels, out_channels, kernel=(1, 1),
#                     stride=(1, 1), pad=(0, 0), with_bias=False),
#             Mo.BatchNormalization(n_features=out_channels, n_dims=4),
#             Mo.ReLU6()
#         )

#     def call(self, input):
#         return self._operators(input)

#     def extra_repr(self):
#         return (f'in_channels={self._in_channels}, '
#                 f'out_channels={self._out_channels}')


# def make_divisible(x, divisible_by=8):
#     return int(np.ceil(x * 1. / divisible_by) * divisible_by)


# class InvertedResidual(Mo.Module):
#     r"""Inverted Residual layer.
#     Args:
#         in_channels([type]): [description]
#         out_channels([type]): [description]
#         stride([type]): [description]
#         expand_ratio([type]): [description]
#     """

#     def __init__(self, in_channels, out_channels, stride, expand_ratio,
#                  kernel=(3, 3), pad=(1, 1)):

#         assert stride in [1, 2]

#         self._stride = stride
#         self._in_channels = in_channels
#         self._out_channels = out_channels
#         self._expand_ratio = expand_ratio

#         hidden_dim = int(in_channels * expand_ratio)
#         self._use_res_connect = (self._stride == 1 and
#                                  in_channels == out_channels)

#         if expand_ratio == 1:
#             self._conv = Mo.Sequential(
#                 # dw
#                 Mo.Conv(hidden_dim, hidden_dim, kernel=kernel,
#                         stride=(stride, stride), pad=pad, group=hidden_dim,
#                         with_bias=False),
#                 Mo.BatchNormalization(n_features=hidden_dim, n_dims=4),
#                 Mo.ReLU6(),
#                 # pw-linear
#                 Mo.Conv(hidden_dim, out_channels, kernel=(1, 1), stride=(1, 1),
#                         with_bias=False),
#                 Mo.BatchNormalization(n_features=out_channels, n_dims=4)
#             )
#         else:
#             self._conv = Mo.Sequential(
#                 # pw
#                 Mo.Conv(in_channels, hidden_dim, kernel=(1, 1), stride=(1, 1),
#                         with_bias=False),
#                 Mo.BatchNormalization(n_features=hidden_dim, n_dims=4),
#                 Mo.ReLU6(),
#                 # dw
#                 Mo.Conv(
#                     hidden_dim, hidden_dim, kernel=kernel,
#                     stride=(stride, stride), pad=pad,
#                     group=hidden_dim, with_bias=False),
#                 Mo.BatchNormalization(n_features=hidden_dim, n_dims=4),
#                 Mo.ReLU6(),
#                 # pw-linear
#                 Mo.Conv(hidden_dim, out_channels, kernel=(1, 1), stride=(1, 1),
#                         with_bias=False),
#                 Mo.BatchNormalization(n_features=out_channels, n_dims=4)
#             )

#     def call(self, x):
#         if self._use_res_connect:
#             return x + self._conv(x)
#         else:
#             return self._conv(x)

#     def extra_repr(self):
#         return (f'in_channels={self._in_channels}, '
#                 f'out_channels={self._out_channels}, '
#                 f'expand_ratio={self._expand_ratio}')
