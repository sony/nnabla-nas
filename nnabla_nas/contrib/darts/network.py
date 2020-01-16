from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F

from ... import module as Mo
from ... import utils as ut
from ...module import MixedOp
from .modules import StemConv

OPS = {
    0: lambda channels, stride, affine: Mo.DilConv(channels, channels, (3, 3), pad=(2, 2), stride=(stride, stride), affine=affine),
    1: lambda channels, stride, affine: Mo.DilConv(channels, channels, (5, 5), pad=(4, 4), stride=(stride, stride), affine=affine),
    2: lambda channels, stride, affine: Mo.SepConv(channels, channels, (3, 3), pad=(1, 1), stride=(stride, stride), affine=affine),
    3: lambda channels, stride, affine: Mo.SepConv(channels, channels, (5, 5), pad=(2, 2), stride=(stride, stride), affine=affine),
    4: lambda channels, stride, affine: Mo.MaxPool(kernel=(3, 3), stride=(stride, stride), pad=(1, 1)),
    5: lambda channels, stride, affine: Mo.AvgPool(kernel=(3, 3), stride=(stride, stride), pad=(1, 1)),
    6: lambda channels, stride, affine: Mo.FactorizedReduce(channels, channels, affine=affine) if stride > 1 else Mo.Identity(),
    7: lambda channels, stride, affine: Mo.Zero((stride, stride))
}


class Cell(Mo.Module):
    def __init__(self, channels, reductions, genotype):
        super().__init__()

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

        cell_type = 'reduce' if reductions[-1] else 'normal'
        cell_arch = genotype[cell_type + '_alpha']

        # build choice blocks
        self._indices = list()
        self._blocks = Mo.ModuleList()
        for i in range(len(cell_arch)):
            for (op_idx, choice_idx) in cell_arch[str(i + 2)]:
                stride = 2 if reductions[-1] and choice_idx < 2 else 1
                self._blocks.add_module(OPS[op_idx](channels[2], stride, True))
                self._indices.append(choice_idx)

    def __call__(self, *input):
        """Each cell has two inputs and one output."""
        out = [op(x) for op, x in zip(self._prep, input)]
        for i in range(len(self._indices) // 2):
            idx = (self._indices[2*i], self._indices[2*i + 1])
            ops = (self._blocks[2*i], self._blocks[2*i + 1])

            choice = list()
            for j, op in zip(idx, ops):
                choice.append(op(out[j]))
                if self.training and not isinstance(op, Mo.Identity):
                    choice[-1] = ut.drop_path(choice[-1])

            out.append(F.add2(choice[0], choice[1]))

        return F.concatenate(*out[2:], axis=1)


class AuxiliaryHeadCIFAR(Mo.Module):
    # NOTE: Already tested (same as pytorch).
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
                 num_ops=8, auxiliary=True, genotype=None):
        super().__init__()
        self._num_ops = num_ops
        self._multiplier = multiplier
        self._init_channels = init_channels
        self._num_cells = num_cells
        self._auxiliary = auxiliary

        num_channels = stem_multiplier * init_channels

        # initialize the arch parameters
        self._stem = StemConv(3, num_channels)
        self._cells = self._init_cells(num_cells, num_channels, genotype)
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
            out_p, out_c = out_c, cell(out_p, out_c)
            if i == 2 * self._num_cells//3:
                if self.training and self._auxiliary:
                    logits_aux = self._auxiliary_head(out_c)
        out_c = self._ave_pool(out_c)
        logits = self._linear(out_c)
        return logits, logits_aux

    def _init_cells(self, num_cells, channel_c, genotype):
        cells = Mo.ModuleList()
        channel_p_p, channel_p, channel_c = channel_c, channel_c, self._init_channels
        reduction_p, reduction_c = False, False

        for i in range(num_cells):
            reduction_c = i in (num_cells // 3, 2 * num_cells // 3)
            channel_c *= reduction_c + 1
            cells.add_module(
                Cell(channels=(channel_p_p, channel_p, channel_c),
                     reductions=(reduction_p, reduction_c),
                     genotype=genotype)
            )
            reduction_p = reduction_c
            channel_p_p, channel_p = channel_p, self._multiplier * channel_c
            if i == 2*num_cells//3:
                self._c_auxiliary = channel_p

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

    def get_arch_modues(self):
        ans = []
        for name, module in self.get_modules():
            if isinstance(module, MixedOp):
                ans.append(module)
        return ans
