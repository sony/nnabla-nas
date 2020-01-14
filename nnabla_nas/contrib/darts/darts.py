from collections import OrderedDict

from nnabla.initializer import ConstantInitializer

from ... import module as Mo
from ...module import MixedOp
from .modules import Cell, StemConv


class Darts(Mo.Model):
    def __init__(self, shape, init_channels, num_cells, num_classes,
                 num_choices=4, multiplier=4, stem_multiplier=3,
                 num_ops=8, shared_params=True, mode='full', drop_prob=None):
        super().__init__()
        self._num_choices = num_choices
        self._num_ops = num_ops
        self._multiplier = multiplier
        self._init_channels = init_channels
        self._mode = mode
        self._drop_prob = drop_prob

        num_channels = stem_multiplier * init_channels

        # initialize the arch parameters
        self._alpha_normal = self._init_alpha(num_ops, shared_params)
        self._alpha_reduce = self._init_alpha(num_ops, shared_params)
        self._stem = StemConv(3, num_channels)
        self._cells = self._init_cells(num_cells, num_channels)
        self._ave_pool = Mo.AvgPool(kernel=(8, 8))
        self._linear = Mo.Linear(self._last_channels, num_classes)

        # save input and output shapes
        self._input_shape = shape
        self._output_shape = (shape[0], num_classes)

    def __call__(self, input):
        out_p = out_c = self._stem(input)
        for cell in self._cells:
            out_c, out_p = cell(out_p, out_c), out_c
        out_c = self._ave_pool(out_c)
        return self._linear(out_c)

    def _init_cells(self, num_cells, channel_c):
        cells = Mo.ModuleList()

        channel_p_p, channel_p, channel_c = channel_c, channel_c, self._init_channels
        reduction_p, reduction_c = False, False

        for i in range(num_cells):
            reduction_c = i in (num_cells // 3, 2 * num_cells // 3)
            channel_c *= reduction_c + 1
            cells.add_module(
                Cell(num_choices=self._num_choices,
                     multiplier=self._multiplier,
                     channels=(channel_p_p, channel_p, channel_c),
                     reductions=(reduction_p, reduction_c),
                     mode=self._mode,
                     alpha=self._alpha_reduce if reduction_c
                     else self._alpha_normal)
            )
            reduction_p = reduction_c
            channel_p_p, channel_p = channel_p, self._multiplier * channel_c

        # save the last channels for the last module
        self._last_channels = channel_p

        return cells

    def _init_alpha(self, n, shared_params):
        alpha = []
        for i in range(self._num_choices):
            for _ in range(i + 2):
                alpha_shape = (n,) + (1, 1, 1, 1)
                alpha_init = ConstantInitializer(0.0)
                alpha.append(Mo.Parameter(alpha_shape, initializer=alpha_init)
                             if shared_params else None)
        return alpha

    def get_net_parameters(self, grad_only=False):
        param = OrderedDict()
        for key, val in self.get_parameters(grad_only).items():
            if '_alpha' not in key:
                param[key] = val
        return param

    def get_arch_parameters(self, grad_only=False):
        param = OrderedDict()
        for key, val in self.get_parameters(grad_only).items():
            if '_alpha' in key:
                param[key] = val
        return param

    def get_arch_modues(self):
        ans = []
        for name, module in self.get_modules():
            if isinstance(module, MixedOp):
                ans.append(module)
        return ans
