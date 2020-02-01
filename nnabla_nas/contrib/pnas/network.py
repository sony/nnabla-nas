from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F

from ... import module as Mo
from ..misc import AuxiliaryHeadCIFAR, DropPath, MixedOp
from . import modules as pnas


class SearchNet(Mo.Module):
    r"""SearchNet for PNAS."""

    def __init__(self, in_channels, init_channels, num_cells, num_classes,
                 num_choices=4, multiplier=4, stem_multiplier=3):
        super().__init__()
        self._num_choices = num_choices
        self._num_ops = len(pnas.CANDIDATE_FUNC)
        self._multiplier = multiplier
        self._init_channels = init_channels

        num_channels = stem_multiplier * init_channels

        # build the network
        self._stem = pnas.StemConv(in_channels, num_channels)
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
                pnas.Cell(
                    num_choices=self._num_choices,
                    multiplier=self._multiplier,
                    channels=(channel_p_p, channel_p, channel_c),
                    reductions=(reduction_p, reduction_c),
                    affine=False
                )
            )
            reduction_p = reduction_c
            channel_p_p, channel_p = channel_p, self._multiplier * channel_c

        # save the last channels for the last module
        self._last_channels = channel_p

        return cells

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


class TrainNet(Mo.Module):
    def __init__(self, in_channels, init_channels, num_cells, num_classes,
                 genotype, num_choices=4, multiplier=4, stem_multiplier=3,
                 drop_path=0.2, num_ops=8, auxiliary=True):
        super().__init__()
        self._num_choices = num_choices
        self._num_ops = num_ops
        self._multiplier = multiplier
        self._init_channels = init_channels
        self._auxiliary = auxiliary
        self._num_cells = num_cells
        self._drop_path = drop_path

        num_channels = stem_multiplier * init_channels

        # initialize the arch parameters
        self._stem = pnas.StemConv(in_channels, num_channels)
        self._cells = self._init_cells(num_cells, num_channels)
        self._ave_pool = Mo.AvgPool(kernel=(8, 8))
        self._linear = Mo.Linear(self._last_channels, num_classes)

        # auxiliary head
        if auxiliary:
            self._auxiliary_head = AuxiliaryHeadCIFAR(
                self._c_auxiliary, num_classes)

        # load weights
        nn.load_parameters(genotype)
        self.set_parameters(nn.get_parameters())
        for key, module in self.get_modules():
            if isinstance(module, MixedOp):
                module._mode = 'max'
                module._update_active_idx()
                module = module._ops[module._active]

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

    def _init_cells(self, num_cells, channel_c):
        cells = Mo.ModuleList()

        channel_p_p, channel_p, channel_c = channel_c, channel_c, \
            self._init_channels
        reduction_p, reduction_c = False, False

        for i in range(num_cells):
            reduction_c = i in (num_cells // 3, 2 * num_cells // 3)
            channel_c *= reduction_c + 1
            cells.append(
                Cell(num_choices=self._num_choices,
                     multiplier=self._multiplier,
                     channels=(channel_p_p, channel_p, channel_c),
                     reductions=(reduction_p, reduction_c),
                     drop_path=self._drop_path,
                     affine=True)
            )
            reduction_p = reduction_c
            channel_p_p, channel_p = channel_p, self._multiplier * channel_c
            if i == 2*num_cells//3:
                self._c_auxiliary = channel_p

        # save the last channels for the last module
        self._last_channels = channel_p

        return cells


class Cell(Mo.Module):
    """Cell in DARTS.
    """

    def __init__(self, num_choices, multiplier, channels, reductions,
                 drop_path=0.2, affine=False):
        super().__init__()
        self._multiplier = multiplier
        self._num_choices = num_choices
        self._channels = channels
        self._affine = affine
        self._drop_path = drop_path

        # preprocess the inputs
        self._prep = Mo.ModuleList()
        if reductions[0]:
            self._prep.append(
                pnas.FactorizedReduce(channels[0], channels[2], affine=affine))
        else:
            self._prep.append(
                pnas.ReLUConvBN(channels[0], channels[2], kernel=(1, 1),
                                affine=affine))
        self._prep.append(
            pnas.ReLUConvBN(channels[1], channels[2], kernel=(1, 1), affine=affine))

        # build choice blocks
        self._blocks = Mo.ModuleList()
        for i in range(num_choices):
            for j in range(i + 2):
                self._blocks.append(
                    pnas.ChoiceBlock(
                        in_channels=channels[2],
                        out_channels=channels[2],
                        is_reduced=j < 2 and reductions[1],
                        mode='sample',
                        affine=affine
                    )
                )

    def call(self, *input):
        """Each cell has two inputs and one output."""
        out = [op(x) for op, x in zip(self._prep, input)]
        offset = 0
        for _ in range(self._num_choices):
            out_temp = []
            for j, h in enumerate(out):
                b = self._blocks[offset + j]
                temp = b(h)
                if self.training and not isinstance(b._mixed, Mo.Identity):
                    temp = DropPath(self._drop_path)(temp)
                out_temp.append(temp)
            s = sum(out_temp)
            offset += len(out)
            out.append(s)
        return F.concatenate(*out[-self._multiplier:], axis=1)

    def __extra_repr__(self):
        return (f'num_choices={self._num_choices}, '
                f'multiplier={self._multiplier}, '
                f'channels={self._channels}, '
                f'mode={self._mode}, '
                f'affine={self._affine}')
