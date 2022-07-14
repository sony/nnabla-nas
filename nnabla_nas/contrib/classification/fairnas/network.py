from collections import OrderedDict

import nnabla.functions as F
import numpy as np
import os

from .... import module as Mo
from ..base import ClassificationModel as Model
from .modules import ChoiceBlock
from ..mobilenet.modules import ConvBNReLU, CANDIDATES
from ..mobilenet.helper import visualize_mobilenet_arch
from ..mobilenet.network import _make_divisible, label_smoothing_loss


class SearchNet(Model):
    r"""MobileNet V2 search space.

    This implementation is based on the PyTorch implementation.

    Args:
        num_classes (int): Number of classes
        width_mult (float, optional): Width multiplier - adjusts number of
            channels in each layer by this amount
        settings (list, optional): Network structure. Defaults to None.
        drop_rate (float, optional): Drop rate used in Dropout. Defaults to 0.
        candidates (list of str, optional): A list of candicates. Defaults to
            None.
        skip_connect (bool, optional): Whether the skip connect is used.
            Defaults to `True`.
        weight (str, optional): The path to weight file. Defaults to
            None.
        seed (int, optional): The seed for the random generator.

    References:
        Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.C., 2018.
            Mobilenetv2: Inverted residuals and linear bottlenecks. In
            Proceedings of the IEEE conference on computer vision and pattern
            recognition (pp. 4510-4520).
    """

    def __init__(self,
                 num_classes=1000,
                 width_mult=1,
                 settings=None,
                 drop_rate=0,
                 candidates=None,
                 skip_connect=True,
                 weights=None,
                 seed=123):

        self._num_classes = num_classes
        self._width_mult = width_mult
        self._skip_connect = skip_connect
        self.rng = np.random.RandomState(seed)
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
        features += [CANDIDATES['MB1 3x3'](
            in_channels, first_cell_width, 1, '/init_block')]
        in_channels = first_cell_width

        if settings is None:
            settings = [
                # c, n, s
                [24, 4, 2],
                [32, 4, 2],
                [64, 4, 2],
                [96, 4, 1],
                [160, 4, 2],
                [320, 1, 1]
            ]
        self._settings = settings
        if candidates is None:
            candidates = [
                "MB3 3x3",
                "MB6 3x3",
                "MB3 5x5",
                "MB6 5x5",
                "MB3 7x7",
                "MB6 7x7"
            ]
        self._candidates = candidates
        # building inverted residual blocks
        for c, n, s in settings:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                curr_candidates = candidates.copy()
                if stride == 1 and in_channels == output_channel \
                        and skip_connect:
                    curr_candidates.append('skip_connect')
                features.append(
                    ChoiceBlock(in_channels, output_channel,
                                stride=stride, ops=curr_candidates, rng=self.rng)
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

        if weights is not None:
            self.load_parameters(weights)

    def get_net_parameters(self, grad_only=False):
        r"""Returns an `OrderedDict` containing architecture parameters.

        Args:
            grad_only (bool, optional): If sets to `True`, then only parameters
                with `need_grad=True` are returned. Defaults to False.

        Returns:
            OrderedDict: A dictionary containing parameters.
        """
        p = self.get_parameters(grad_only)
        return OrderedDict([(k, v) for k, v in p.items() if 'alpha' not in k])

    def call(self, input):
        out = self._features(input)
        return self._classifier(out)

    def extra_repr(self):
        return (f'num_classes={self._num_classes}, '
                f'width_mult={self._width_mult}, '
                f'settings={self._settings}, '
                f'candidates={self._candidates}, '
                f'skip_connect={self._skip_connect}')

    def visualize(self, path):
        # save the architectures
        if isinstance(self._features[2]._mixed, Mo.MixedOp):
            visualize_mobilenet_arch(self, os.path.join(path, 'arch'))

    def loss(self, outputs, targets, loss_weights=None):
        assert len(outputs) == 1 and len(targets) == 1
        return F.mean(label_smoothing_loss(outputs[0], targets[0]))

    def get_arch(self):
        self.arch = []
        for _, module in self.get_modules():
            if isinstance(module, ChoiceBlock):
                idx = module._mixed._active
                self.arch.append(idx)
        return self.arch


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
        skip_connect (bool, optional): Whether the skip connect is used.
            Defaults to `True`.
        genotype(str, optional): The path to architecture file. Defaults to
            None.

    References:
        Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.C., 2018.
            Mobilenetv2: Inverted residuals and linear bottlenecks. In
            Proceedings of the IEEE conference on computer vision and pattern
            recognition (pp. 4510-4520).
    """

    def __init__(self,
                 num_classes=1000,
                 width_mult=1,
                 settings=None,
                 drop_rate=0,
                 candidates=None,
                 skip_connect=True,
                 genotype=None,
                 weights=None):

        super().__init__(num_classes=num_classes, width_mult=width_mult,
                         settings=settings, drop_rate=drop_rate,
                         candidates=candidates,
                         skip_connect=skip_connect, weights=weights)

        if genotype is not None:
            arch = iter(genotype)
            for _, module in self.get_modules():
                if isinstance(module, ChoiceBlock):
                    idx = next(arch)
                    module._mixed = module._mixed._ops[idx]
