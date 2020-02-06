import nnabla.functions as F

from .. import module as Mo


class Model(Mo.Module):

    def get_net_parameters(self, grad_only=False):
        raise NotImplementedError

    def get_arch_parameters(self, grad_only=False):
        raise NotImplementedError

    def sample(self):
        pass

    def arch_modules():
        pass


class AuxiliaryHeadCIFAR(Mo.Module):
    r"""Auxiliary head used for CIFAR10 dataset.

    Args:
        channels (:obj:`int`): The number of input channels.
        num_classes (:obj:`int`): The number of classes.
    """

    def __init__(self, channels, num_classes):
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

    def extra_repr(self):
        return f'channels={self._channels}, num_classes={self._num_classes}'


class DropPath(Mo.Module):
    r"""Drop Path layer.

    Args:
        drop_prob (:obj:`int`, optional): The probability of droping path.
            Defaults to 0.2.
    """

    def __init__(self, drop_prob=0.2):
        self._drop_prob = drop_prob

    def call(self, input):
        if self._drop_prob == 0:
            return input
        mask = F.rand(shape=(input.shape[0], 1, 1, 1))
        mask = F.greater_equal_scalar(mask, self._drop_prob)
        out = F.mul_scalar(input, 1. / (1 - self._drop_prob))
        out = F.mul2(out, mask)
        return out

    def extra_repr(self):
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
    """

    def __init__(self, in_channels, out_channels, kernel,
                 pad=None, stride=None):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel
        self._pad = pad
        self._stride = stride

        self._operators = Mo.Sequential(
            Mo.ReLU(),
            Mo.Conv(in_channels, out_channels, kernel=kernel,
                    stride=stride, pad=pad, with_bias=False),
            Mo.BatchNormalization(n_features=out_channels, n_dims=4)
        )

    def call(self, input):
        return self._operators(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'pad={self._pad}')


class SepConv(Mo.DwConv):
    def __init__(self, out_channels, *args, **kwargs):
        Mo.DwConv.__init__(self, *args, **kwargs)
        self._out_channels = out_channels
        self._conv_module_pw = Mo.Conv(self._in_channels, out_channels,
                                       kernel=(1, 1), pad=None, group=1,
                                       rng=self._rng, with_bias=False)

    def call(self, input):
        return self._conv_module_pw(Mo.DwConv.call(self, input))
