from collections import OrderedDict

import numpy as np

from ... import module as Mo
from ...utils import load_parameters
from ..model import Model
from .modules import ChoiceBlock
from .modules import ConvBNReLU
from .modules import InvertedResidual


def _make_divisible(x, divisible_by=8):
    r"""It ensures that all layers have a channel number that is divisible by
    divisible_by."""
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class SearchNet(Model):
    r"""MobileNet V2.

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
        block: Module specifying inverted residual building block for
            mobilenet
    """

    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 settings=None,
                 round_nearest=8,
                 block=None,
                 mode='full'):

        self._num_classes = num_classes
        self._width_mult = width_mult
        self._round_nearest = round_nearest

        block = block or InvertedResidual
        in_channels = 32
        last_channel = 1280
        n_max = 4

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
            n_iter = n_max if k > 0 else 1
            for i in range(n_iter):
                stride = s if i == 0 else 1
                # if k == 0:
                #     block(in_channels, output_channel, stride, expand_ratio=1)
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

        self.set_parameters(load_parameters(genotype))
        for _, module in self.get_modules():
            if isinstance(module, ChoiceBlock):
                idx = np.argmax(module._mixed._alpha.d)
                module._mixed = module._mixed._ops[idx]

        # def _make_divisible(v, divisor, min_value=None):
        #     r"""This function is taken from the original tf repo.
        #     It ensures that all layers have a channel number that is divisible by 8.
        #     """
        #     if min_value is None:
        #         min_value = divisor
        #     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        #     # Make sure that round down does not go down by more than 10%.
        #     if new_v < 0.9 * v:
        #         new_v += divisor
        #     return new_v

        # class MobileNetV2(Mo.Module):
        #     """MobileNetV2

        #     Args:
        #         Mo ([type]): [description]
        #         n_class (int, optional): [description]. Defaults to 10.
        #         input_size (int, optional): [description]. Defaults to 32.
        #         width_mult ([type], optional): [description]. Defaults to 1..
        #     """

        #     def __init__(self, n_class=10, input_size=32, width_mult=1.):

        #         block = InvertedResidual
        #         input_channel = 32
        #         last_channel = 1280
        #         settings = [
        #             # t, c, n, s
        #             [1, 16, 1, 1],
        #             [6, 24, 2, 1],
        #             [6, 32, 3, 1],
        #             [6, 64, 4, 2],
        #             [6, 96, 3, 1],
        #             [6, 160, 3, 2],
        #             [6, 320, 1, 1],
        #         ]

        #         # building first layer
        #         assert input_size % 32 == 0
        #         # input_channel = make_divisible(input_channel * width_mult)
        #         # first channel is always 32!
        #         self.last_channel = (make_divisible(last_channel * width_mult)
        #                              if width_mult > 1.0 else last_channel)
        #         self.features = Mo.Sequential()
        #         self.features.append(ConvBN(3, input_channel, (2, 2)))

        #         # building inverted residual blocks
        #         for t, c, n, s in settings:
        #             output_channel = make_divisible(c * width_mult) if t > 1 else c
        #             for i in range(4):
        #                 if i == 0:
        #                     self.features.append(block(input_channel, output_channel,
        #                                                s, expand_ratio=t))
        #                 else:
        #                     self.features.append(block(input_channel, output_channel,
        #                                                1, expand_ratio=t))
        #                 input_channel = output_channel
        #         # building last several layers
        #         self.features.append(Conv1x1BN(input_channel, self.last_channel))
        #         self.features.append(Mo.AvgPool((4, 4)))
        #         self.features.append(Mo.Dropout(0.2))
        #         # building classifier
        #         self.classifier = Mo.Linear(self.last_channel, n_class)

        #     def call(self, x):
        #         x = self.features(x)
        #         x = self.classifier(x)
        #         return x
