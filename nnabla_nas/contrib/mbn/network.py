from collections import Counter, OrderedDict

import numpy as np

from ... import module as Mo
from ...utils import load_parameters
from ..model import Model
from .modules import CANDIDATES, ChoiceBlock, ConvBNReLU


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
        drop_rate (float, optional): Drop rate used in Dropout. Defaults to 0.
        candidates (list of str, optional): A list of candicates. Defaults to
            None.
        skip_connect (bool, optional): Whether the skip connect is used.
            Defaults to `True`.

    References:
    [1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.C., 2018.
        Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings
        of the IEEE conference on computer vision and pattern recognition
        (pp. 4510-4520).
    """
    """[summary]
    #  n_cell_stages=(4, 4, 4, 4, 4, 1),
                #  width_stages=(24, 40, 80, 96, 192, 320),
                #  stride_stages=(2, 2, 2, 1, 2, 1),
    Returns:
        [type]: [description]
    """

    def __init__(self,
                 num_classes=1000,
                 width_mult=1,
                 settings=None,
                 drop_rate=0,
                 candidates=None,
                 mode='sample',
                 skip_connect=True):

        self._num_classes = num_classes
        self._width_mult = width_mult
        round_nearest = 8

        in_channels = 32
        last_channel = 1280

        # building first layer
        in_channels = _make_divisible(in_channels * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult),
            round_nearest
        )
        features = [ConvBNReLU(3, in_channels, stride=(2, 2))]

        first_cell_width = _make_divisible(16 * width_mult, 8)
        features += [CANDIDATES['InvertedResidual_t1_k3'](
            in_channels, first_cell_width, 1)]
        in_channels = first_cell_width

        if settings is None:
            settings = [
                # c, n, s
                [24, 4, 1],
                [32, 4, 1],
                [64, 4, 2],
                [96, 4, 1],
                [160, 4, 2],
                [320, 1, 1]
            ]

        if candidates is None:
            candidates = [
                "InvertedResidual_t3_k3",
                "InvertedResidual_t6_k3",
                "InvertedResidual_t3_k5",
                "InvertedResidual_t6_k5",
                "InvertedResidual_t3_k7",
                "InvertedResidual_t6_k7"
            ]

        # building inverted residual blocks
        for c, n, s in settings:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                curr_candidates = candidates.copy()
                if (stride == 1 and in_channels == output_channel
                        and skip_connect):
                    curr_candidates.append('skip_connect')
                features.append(
                    ChoiceBlock(in_channels, output_channel,
                                stride=stride, mode=mode,
                                ops=curr_candidates)
                )
                in_channels = output_channel

        # building last several layers
        features.append(ConvBNReLU(in_channels, self.last_channel,
                                   kernel=(1, 1)))
        # make it nn.Sequential
        self._features = Mo.Sequential(*features)

        # building classifier
        self._classifier = Mo.Sequential(
            Mo.GlobalAvgPool(),
            Mo.Dropout(drop_rate),
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

    def summary(self):
        stats = []
        arch_params = self.get_arch_parameters()
        count = Counter([np.argmax(m.d.flat) for m in arch_params.values()])
        op_names = [
            'InvertedResidual_t3_k3', 'InvertedResidual_t6_k3',
            'InvertedResidual_t3_k5', 'InvertedResidual_t6_k5',
            'InvertedResidual_t3_k7', 'InvertedResidual_t6_k7',
            'skip_connect'
        ]
        total = len(arch_params)
        for k in range(len(op_names)):
            name = op_names[k]
            stats.append(name + f' = {count[k]/total*100:.2f}%\t')
        return ''.join(stats)


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
        skip_connect (bool, optional): Whether the skip connect is used.
            Defaults to `True`.
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
                 width_mult=1,
                 settings=None,
                 drop_rate=0,
                 candidates=None,
                 mode='sample',
                 skip_connect=True,
                 genotype=None):

        super().__init__(num_classes=num_classes, width_mult=width_mult,
                         settings=settings, drop_rate=drop_rate,
                         candidates=candidates, mode=mode,
                         skip_connect=skip_connect)

        if genotype is not None:
            self.set_parameters(load_parameters(genotype))
            for _, module in self.get_modules():
                if isinstance(module, ChoiceBlock):
                    idx = np.argmax(module._mixed._alpha.d)
                    module._mixed = module._mixed._ops[idx]
