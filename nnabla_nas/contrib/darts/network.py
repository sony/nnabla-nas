from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
from nnabla.initializer import ConstantInitializer

from ... import module as Mo
from ... import utils as ut
from ...module import MixedOp
from .modules import ChoiceBlock, StemConv


class Cell(Mo.Module):
    def __init__(self, num_choices, multiplier, channels, reductions,
                 mode='full', alpha=None, drop_prob=None):
        super().__init__()
        self._multiplier = multiplier
        self._num_choices = num_choices
        self._drop_prob = drop_prob
        # preprocess the inputs
        self._prep = Mo.ModuleList()
        if reductions[0]:
            self._prep.add_module(
                Mo.FactorizedReduce(channels[0], channels[2]))
        else:
            self._prep.add_module(Mo.ReLUConvBN(
                channels[0], channels[2], kernel=(1, 1)))
        self._prep.add_module(Mo.ReLUConvBN(
            channels[1], channels[2], kernel=(1, 1)))
        # build choice blocks
        self._blocks = Mo.ModuleList()
        for i in range(num_choices):
            for j in range(i + 2):
                self._blocks.add_module(
                    ChoiceBlock(in_channels=channels[2],
                                out_channels=channels[2],
                                is_reduced=j < 2 and reductions[1],
                                mode=mode,
                                alpha=alpha[len(self._blocks)])
                )

    def __call__(self, *input):
        """Each cell has two inputs and one output."""
        out = [op(x) for op, x in zip(self._prep, input)]
        offset = 0
        for _ in range(self._num_choices):
            aux = []
            for j, h in enumerate(out):
                op = self._blocks[offset + j]
                idx = op._mixed._active
                if idx != 7:  # check if it's zero op
                    x = op(h)
                    # if it's not identity op
                    if self._drop_prob is not None and (op._is_reduced or idx != 6):
                        x = ut.drop_path(x, self._drop_prob)
                    aux.append(x)
            s = sum(aux)
            offset += len(out)
            out.append(s)
        return F.concatenate(*out[-self._multiplier:], axis=1)


class AuxiliaryHeadCIFAR(Mo.Module):

    def __init__(self, channels, num_classes):
        super().__init__()
        self.feature = Mo.Sequential(
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
        self.classifier = Mo.Linear(in_features=768, out_features=num_classes)

    def __call__(self, input):
        out = self.feature(input)
        return self.classifier(out)


class NetworkCIFAR(Mo.Model):
    def __init__(self, shape, init_channels, num_cells, num_classes,
                 num_choices=4, multiplier=4, stem_multiplier=3,
                 num_ops=8, shared_params=True, mode='full',
                 drop_prob=0.2, auxiliary=True):
        super().__init__()
        self._num_choices = num_choices
        self._num_ops = num_ops
        self._multiplier = multiplier
        self._init_channels = init_channels
        self._mode = mode
        self._drop_prob = nn.Variable(
            (1, 1, 1, 1), need_grad=False) if drop_prob > 0 else None
        self._num_cells = num_cells
        self._auxiliary = auxiliary

        num_channels = stem_multiplier * init_channels

        # initialize the arch parameters
        self._alpha_normal = self._init_alpha(num_ops, shared_params)
        self._alpha_reduce = self._init_alpha(num_ops, shared_params)
        self._stem = StemConv(3, num_channels)
        self._cells = self._init_cells(num_cells, num_channels)
        self._ave_pool = Mo.AvgPool(kernel=(8, 8))
        self._linear = Mo.Linear(self._last_channels, num_classes)

        # auxiliary head
        self._auxiliary_head = AuxiliaryHeadCIFAR(
            self._c_auxiliary, num_classes)
        # save input and output shapes
        self._input_shape = shape
        self._output_shape = (shape[0], num_classes)

    def __call__(self, input):
        logits_aux = None
        out_p = out_c = self._stem(input)
        for i, cell in enumerate(self._cells):
            out_c, out_p = cell(out_p, out_c), out_c
            if i == 2 * self._num_cells//3:
                if self.training and self._auxiliary:
                    logits_aux = self._auxiliary_head(out_c)
        out_c = self._ave_pool(out_c)
        return self._linear(out_c), logits_aux

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
                     else self._alpha_normal,
                     drop_prob=self._drop_prob)
            )
            reduction_p = reduction_c
            channel_p_p, channel_p = channel_p, self._multiplier * channel_c
            if i == 2*num_cells//3:
                self._c_auxiliary = channel_p

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
