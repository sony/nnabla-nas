import json
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
from nnabla.initializer import ConstantInitializer

from ... import module as Mo
from ..misc import AuxiliaryHeadCIFAR
from ..misc import DropPath
from ..misc import Model
from . import modules as darts


class SearchNet(Model):
    r"""SearchNet for DARTS."""

    def __init__(self, in_channels, init_channels, num_cells, num_classes,
                 num_choices=4, multiplier=4, mode='full', shared=False,
                 stem_multiplier=3):
        self._in_channels = in_channels
        self._init_channels = init_channels
        self._num_cells = num_cells
        self._num_classes = num_classes
        self._num_choices = num_choices
        self._multiplier = multiplier
        self._mode = mode
        self._shared = shared

        num_channels = stem_multiplier * init_channels

        # initialize the arch parameters
        self._alpha = self._init_alpha() if shared else None

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

    def _init_cells(self, num_cells, C):
        cells = Mo.ModuleList()
        Cpp, Cp, C = C, C, self._init_channels
        reduction_p, reduction_c = False, False
        for i in range(num_cells):
            reduction_c = i in (num_cells // 3, 2 * num_cells // 3)
            C *= reduction_c + 1
            cells.append(
                darts.Cell(
                    num_choices=self._num_choices,
                    multiplier=self._multiplier,
                    channels=(Cpp, Cp, C),
                    reductions=(reduction_p, reduction_c),
                    mode='full',
                    alpha=self._alpha[reduction_c] if self._shared else None
                )
            )
            reduction_p = reduction_c
            Cpp, Cp = Cp, self._multiplier * C
            if i == 2 * num_cells // 3:
                self._c_auxiliary = Cp
        # save the last channels for the last module
        self._last_channels = Cp
        return cells

    def _init_alpha(self):
        r"""Returns a list of parameters."""
        shape = (len(darts.CANDIDATES), 1, 1, 1, 1)
        init = ConstantInitializer(0.0)
        n = self._num_choices * (self._num_choices + 3) // 2
        alpha = Mo.ModuleList()
        for i in range(2):
            params = [Mo.Parameter(shape, initializer=init) for _ in range(n)]
            alpha.append(Mo.ParameterList(params))
        return alpha

    def get_net_parameters(self, grad_only=False):
        p = self.get_parameters(grad_only)
        return OrderedDict([(k, v) for k, v in p.items() if 'alpha' not in k])

    def get_arch_parameters(self, grad_only=False):
        if self._shared:
            return self._alpha.get_parameters()
        p = self.get_parameters(grad_only)
        return OrderedDict([(k, v) for k, v in p.items() if 'alpha' in k])


class TrainNet(Model):
    """TrainNet for DARTS."""

    def __init__(self, in_channels, init_channels, num_cells, num_classes,
                 genotype, num_choices=4, multiplier=4, stem_multiplier=3,
                 drop_path=0.2, auxiliary=True):
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
