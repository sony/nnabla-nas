import numpy as np

from ... import module as Mo
from ...utils import load_parameters
from ..model import Model
from .modules import ChoiceBlock
from .modules import ConvBNReLU
from .modules import InvertedResidual
from .modules import CANDIDATES
from collections import OrderedDict


def _make_divisible(x, divisible_by=8):
    r"""It ensures that all layers have a channel number that is divisible by
    divisible_by."""
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class SearchNet(Model):
    r"""MobileNet V2 search space.

    This implementation is based on the PyTorch implementation.

    Args:
        num_classes (int): Number of classes
        width_mult (float, optional): Width multiplier - adjusts number of
            channels in each layer by this amount
        settings (list, optional): Network structure.
            Defaults to None.
        round_nearest (int, optional): Round the number of channels in
            each layer to be a multiple of this number. Set to 1 to turn
            off rounding.
        n_max (int, optional): The number of blocks. Defaults to 4.
        block: Module specifying inverted residual building block for
            mobilenet

    References:
    [1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.C., 2018.
        Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings
        of the IEEE conference on computer vision and pattern recognition
        (pp. 4510-4520).
    """

    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 settings=None,
                 round_nearest=8,
                 block=None,
                 n_max=4,
                 mode='full'):

        self._num_classes = num_classes
        self._width_mult = width_mult
        self._round_nearest = round_nearest
        self._n_max = n_max

        block = block or InvertedResidual
        in_channels = 32
        last_channel = 1280

        if settings is None:
            settings = [
                # c, s
                [16, 1],
                [24, 1],
                [32, 1],
                [64, 2],
                [96, 1],
                [160, 2],
                [320, 1]
            ]

        # only check the first element
        if len(settings) == 0 or len(settings[0]) != 2:
            raise ValueError(
                "Inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(settings)
            )

        # building first layer
        in_channels = _make_divisible(in_channels * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult),
            round_nearest
        )
        features = [ConvBNReLU(3, in_channels, stride=(2, 2))]

        # output_channel = _make_divisible(16 * width_mult, round_nearest)
        # features.append(block(in_channels, 16, 1, expand_ratio=1))

        # building inverted residual blocks
        for k, (c, s) in enumerate(settings):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            n_iter = n_max
            for i in range(n_iter):
                stride = s if i == 0 else 1
                features.append(
                    ChoiceBlock(in_channels, output_channel,
                                stride=stride, mode=mode, is_skipped=i > 0)
                )
                in_channels = output_channel

        # building last several layers
        features.append(ConvBNReLU(in_channels, self.last_channel,
                                   kernel=(1, 1)))
        # make it nn.Sequential
        self._features = Mo.Sequential(*features)

        # building classifier
        self._classifier = Mo.Sequential(
            Mo.AvgPool((4, 4)),
            Mo.Dropout(0.2),
            Mo.Linear(self.last_channel, num_classes),
        )

    def get_net_parameters(self, grad_only=False):
        r"""Returns an `OrderedDict` containing model parameters.

        Args:
            grad_only (bool, optional): If sets to `True`, then only parameters
                with `need_grad=True` are returned. Defaults to False.

        Returns:
            OrderedDict: A dictionary containing parameters.
        """
        p = self.get_parameters(grad_only)
        return OrderedDict([(k, v) for k, v in p.items() if 'alpha' not in k])

    def get_arch_parameters(self, grad_only=False):
        r"""Returns an `OrderedDict` containing architecture parameters.

        Args:
            grad_only (bool, optional): If sets to `True`, then only parameters
                with `need_grad=True` are returned. Defaults to False.

        Returns:
            OrderedDict: A dictionary containing parameters.
        """
        p = self.get_parameters(grad_only)
        return OrderedDict([(k, v) for k, v in p.items() if 'alpha' in k])

    def call(self, input):
        out = self._features(input)
        return self._classifier(out)

    def extra_repr(self):
        return (f'num_classes={self._num_classes}, '
                f'width_mult={self._width_mult}, '
                f'round_nearest={self._round_nearest}')


class TrainNet(SearchNet):
    r"""MobileNet V2 Train Net.

    Args:
        num_classes (int): Number of classes
        width_mult (float, optional): Width multiplier - adjusts number of
            channels in each layer by this amount
        settings (list, optional): Network structure.
            Defaults to None.
        round_nearest (int, optional): Round the number of channels in
            each layer to be a multiple of this number. Set to 1 to turn
            off rounding.
        n_max (int, optional): The number of blocks. Defaults to 4.
        block: Module specifying inverted residual building block for
            mobilenet. Defaults to None.
        mode (str, optional): The sampling strategy ('full', 'max', 'sample').
            Defaults to 'full'.
        genotype(str, optional): The path to architecture file. Defaults to
            None.

    References:
    [1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.C., 2018.
        Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings
        of the IEEE conference on computer vision and pattern recognition
        (pp. 4510-4520).
    """

    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 settings=None,
                 round_nearest=8,
                 n_max=4,
                 block=None,
                 mode='full',
                 genotype=None):

        super().__init__(num_classes=num_classes, width_mult=width_mult,
                         settings=settings, round_nearest=round_nearest,
                         n_max=n_max, block=block, mode=mode)

        if genotype is not None:
            self.set_parameters(load_parameters(genotype))
            for _, module in self.get_modules():
                if isinstance(module, ChoiceBlock):
                    idx = np.argmax(module._mixed._alpha.d)
                    module._mixed = module._mixed._ops[idx]


class RefNet(SearchNet):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 settings=None,
                 round_nearest=8,
                 block=None,
                 mode='full',
                 genotype=None):

        super().__init__(num_classes=num_classes, width_mult=width_mult,
                         settings=settings, round_nearest=round_nearest,
                         block=block, mode=mode)
        ref_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]]

        arch_idx = self.get_ops_idx(ref_setting)
        i = 0
        for k, v in self.get_arch_parameters().items():
            v.d[arch_idx[i]] = 1.0
            i += 1
        import nnabla as nn
        nn.save_parameters('mbn_ref_arch.h5', self.get_arch_parameters())

    def get_ops_idx(self, setting):
        ops = list()
        for t, c, n, s in setting:
            for m in range(self._n_max):
                ops += [list(CANDIDATES).index(
                        'InvertedResidual_t{}_k3'.format(t))
                        if m < n
                        else list(CANDIDATES).index('skip_connect'.format(t))]
        return ops
