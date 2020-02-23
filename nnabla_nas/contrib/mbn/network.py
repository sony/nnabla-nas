import numpy as np

from ... import module as Mo
from ...utils import load_parameters
from ..model import Model
# from .modules import ChoiceBlock
from .modules import ConvBNReLU
from .modules import CANDIDATES
from collections import OrderedDict
from ..darts.modules import MixedOp
from collections import Counter


def _make_divisible(x, divisible_by=8):
    r"""It ensures that all layers have a channel number that is divisible by
    divisible_by."""
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class ChoiceBlock(Mo.Module):
    def __init__(self, in_channels, out_channels, stride,
                 ops, mode='sample'):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._mode = mode

        self._mixed = MixedOp(
            operators=[func(in_channels, out_channels, stride)
                       for k, func in CANDIDATES.items() if k in ops],
            mode=mode,
        )

    def call(self, input):
        return self._mixed(input)

    def extra_repr(self):
        return (f'in_channels={self._in_channels}, '
                f'out_channels={self._out_channels}, '
                f'stride={self._stride}, '
                f'mode={self._mode}, '
                f'is_skipped={self._is_skipped}')


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
                 width_mult=1,
                 n_cell_stages=(4, 4, 4, 4, 4, 1),
                 width_stages=(24, 40, 80, 96, 192, 320),
                 stride_stages=(2, 2, 2, 1, 2, 1),
                 drop_rate=0,
                 mode='sample'):

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

        # building inverted residual blocks
        default_candidates = [
            'InvertedResidual_t3_k3', 'InvertedResidual_t6_k3',
            'InvertedResidual_t3_k5', 'InvertedResidual_t6_k5',
            'InvertedResidual_t3_k7', 'InvertedResidual_t6_k7'
        ]

        for c, n, s in zip(width_stages, n_cell_stages, stride_stages):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                curr_candidates = default_candidates.copy()
                if stride == 1 and in_channels == output_channel:
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
                 n_cell_stages=(4, 4, 4, 4, 4, 1),
                 width_stages=(24, 40, 80, 96, 192, 320),
                 stride_stages=(2, 2, 2, 1, 2, 1),
                 drop_rate=0,
                 mode='sample',
                 genotype=None):

        super().__init__(num_classes=num_classes, width_mult=width_mult,
                         n_cell_stages=n_cell_stages,
                         width_stages=width_stages,
                         stride_stages=stride_stages, drop_rate=0, mode=mode)

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
