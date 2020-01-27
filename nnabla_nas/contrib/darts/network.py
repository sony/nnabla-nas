import json
from collections import OrderedDict

import nnabla.functions as F
from nnabla.initializer import ConstantInitializer

from ... import module as Mo
from ..misc import AuxiliaryHeadCIFAR, DropPath, MixedOp
from . import modules as darts


class SearchNet(Mo.Module):
    r"""SearchNet for DARTS."""

    def __init__(self, in_channels, init_channels, num_cells, num_classes,
                 num_choices=4, multiplier=4, stem_multiplier=3):
        super().__init__()
        self._num_choices = num_choices
        self._num_ops = len(darts.CANDIDATE_FUNC)
        self._multiplier = multiplier
        self._init_channels = init_channels

        num_channels = stem_multiplier * init_channels

        # initialize the arch parameters
        self._alpha_normal = self._init_alpha(self._num_ops)
        self._alpha_reduce = self._init_alpha(self._num_ops)

        # build the network
        self._stem = darts.StemConv(in_channels, num_channels)
        self._cells = self._init_cells(num_cells, num_channels)
        self._ave_pool = Mo.AvgPool(kernel=(8, 8))
        self._linear = Mo.Linear(self._last_channels, num_classes)

    def call(self, input):
        out_p = out_c = self._stem(input)
        for cell in self._cells:
            out_c, out_p = cell(out_p, out_c), out_c
        out_c = self._ave_pool(out_c)
        return self._linear(out_c)

    def _init_cells(self, num_cells, channel_c):
        cells = Mo.ModuleList()

        channel_p_p, channel_p, channel_c = channel_c, channel_c, \
            self._init_channels
        reduction_p, reduction_c = False, False

        for i in range(num_cells):
            reduction_c = i in (num_cells // 3, 2 * num_cells // 3)
            channel_c *= reduction_c + 1
            cells.append(
                darts.Cell(
                    num_choices=self._num_choices,
                    multiplier=self._multiplier,
                    channels=(channel_p_p, channel_p, channel_c),
                    reductions=(reduction_p, reduction_c),
                    mode='full',
                    alpha=self._alpha_reduce if reduction_c
                    else self._alpha_normal
                )
            )
            reduction_p = reduction_c
            channel_p_p, channel_p = channel_p, self._multiplier * channel_c

        # save the last channels for the last module
        self._last_channels = channel_p

        return cells

    def _init_alpha(self, n):
        alpha = []
        for i in range(self._num_choices):
            for _ in range(i + 2):
                alpha_shape = (n,) + (1, 1, 1, 1)
                alpha_init = ConstantInitializer(0.0)
                alpha.append(Mo.Parameter(alpha_shape, initializer=alpha_init))
        return alpha

    def get_net_parameters(self, grad_only=False):
        param = OrderedDict()
        for key, val in self.get_parameters(grad_only).items():
            if '_alpha' not in key:
                param[key] = val
        return param

    def get_arch_parameters(self, grad_only=False):
        param = OrderedDict()
        for i, alpha in enumerate(self._alpha_normal):
            param['_alpha_normal_{}'.format(i)] = alpha
        for i, alpha in enumerate(self._alpha_reduce):
            param['_alpha_reduce_{}'.format(i)] = alpha
        return param

    def get_arch_modules(self):
        ans = []
        for name, module in self.get_modules():
            if isinstance(module, MixedOp):
                ans.append(module)
        return ans


class TrainNet(Mo.Module):
    """TrainNet for DARTS."""

    def __init__(self, in_channels, init_channels, num_cells, num_classes,
                 genotype, num_choices=4, multiplier=4, stem_multiplier=3,
                 drop_path=0.2, auxiliary=True):
        super().__init__()
        self._num_ops = len(darts.CANDIDATE_FUNC)
        self._multiplier = multiplier
        self._init_channels = init_channels
        self._num_cells = num_cells
        self._auxiliary = auxiliary
        self._num_choices = num_choices
        self._drop_path = drop_path

        num_channels = stem_multiplier * init_channels
        genotype = json.load(open(genotype, 'r'))

        # initialize the arch parameters
        self._stem = darts.StemConv(in_channels, num_channels)
        self._cells = self._init_cells(num_cells, num_channels, genotype)
        self._ave_pool = Mo.AvgPool(kernel=(8, 8))
        self._linear = Mo.Linear(self._last_channels, num_classes)

        # auxiliary head
        if auxiliary:
            self._auxiliary_head = AuxiliaryHeadCIFAR(
                self._c_auxiliary, num_classes)

    def call(self, input):
        logits_aux = None
        out_p = out_c = self._stem(input)
        for i, cell in enumerate(self._cells):
            out_p, out_c = out_c, cell(out_p, out_c)
            if i == 2 * self._num_cells//3:
                if self.training and self._auxiliary:
                    logits_aux = self._auxiliary_head(out_c)
        out_c = self._ave_pool(out_c)
        logits = self._linear(out_c)
        return logits, logits_aux

    def _init_cells(self, num_cells, channel_c, genotype):
        cells = Mo.ModuleList()
        channel_p_p, channel_p, channel_c = channel_c, channel_c,\
            self._init_channels
        reduction_p, reduction_c = False, False

        for i in range(num_cells):
            reduction_c = i in (num_cells // 3, 2 * num_cells // 3)
            channel_c *= reduction_c + 1
            cells.append(
                Cell(channels=(channel_p_p, channel_p, channel_c),
                     reductions=(reduction_p, reduction_c),
                     genotype=genotype,
                     drop_path=self._drop_path)
            )
            reduction_p = reduction_c
            channel_p_p, channel_p = channel_p, self._multiplier * channel_c
            if i == 2*num_cells//3:
                self._c_auxiliary = channel_p

        # save the last channels for the last module
        self._last_channels = channel_p

        return cells

    def get_arch_modues(self):
        ans = []
        for name, module in self.get_modules():
            if isinstance(module, MixedOp):
                ans.append(module)
        return ans


class Cell(Mo.Module):
    def __init__(self, channels, reductions, genotype, drop_path):
        super().__init__()
        self._drop_path = drop_path
        # preprocess the inputs
        self._prep = Mo.ModuleList()
        if reductions[0]:
            self._prep.append(
                darts.FactorizedReduce(channels[0], channels[2]))
        else:
            self._prep.append(darts.ReLUConvBN(
                channels[0], channels[2], kernel=(1, 1)))
        self._prep.append(darts.ReLUConvBN(
            channels[1], channels[2], kernel=(1, 1)))

        cell_type = 'reduce' if reductions[-1] else 'normal'
        cell_arch = genotype[cell_type + '_alpha']

        # build choice blocks
        self._indices = list()
        self._blocks = Mo.ModuleList()
        candidates = list(darts.CANDIDATE_FUNC.values())

        for i in range(len(cell_arch)):
            for (op_idx, choice_idx) in cell_arch[str(i + 2)]:
                stride = 2 if reductions[-1] and choice_idx < 2 else 1
                self._blocks.append(
                    candidates[op_idx](channels[2], stride, True)
                )
                self._indices.append(choice_idx)

    def call(self, *input):
        """Each cell has two inputs and one output."""
        out = [op(x) for op, x in zip(self._prep, input)]
        for i in range(len(self._indices) // 2):
            idx = (self._indices[2*i], self._indices[2*i + 1])
            ops = (self._blocks[2*i], self._blocks[2*i + 1])

            choice = list()
            for j, op in zip(idx, ops):
                choice.append(op(out[j]))
                if self.training and not isinstance(op, Mo.Identity):
                    choice[-1] = DropPath(self._drop_path)(choice[-1])

            out.append(F.add2(choice[0], choice[1]))

        return F.concatenate(*out[2:], axis=1)
