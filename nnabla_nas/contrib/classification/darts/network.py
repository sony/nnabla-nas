# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
from collections import OrderedDict
import json
import os

import nnabla.functions as F
from nnabla.initializer import ConstantInitializer
import numpy as np

from . import modules as darts
from .... import module as Mo
from ..base import ClassificationModel as Model
from ..misc import AuxiliaryHeadCIFAR
from .helper import save_dart_arch, visualize_dart_arch
from hydra import utils


class SearchNet(Model):
    r"""DARTS: Differentiable Architecture Search.

    This is the search space for DARTS.

    Args:
        in_channels (int): The number of input channels.
        init_channels (int): The initial number of channels on each cell.
        num_cells (int): The number of cells.
        num_classes (int): The number of classes.
        num_choices (int, optional): The number of choice blocks on each cell.
            Defaults to 4.
        multiplier (int, optional): The multiplier. Defaults to 4.
        mode (str, optional): The sampling strategy ('full', 'max', 'sample').
            Defaults to 'full'.
        shared (bool, optional): If parameters are shared between cells.
            Defaults to False.
        stem_multiplier (int, optional): The multiplier used for stem
            convolution. Defaults to 3.
    """

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
        self._alpha = self._init_alpha()

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
        """Initializes the cells used in DARTS.

        Args:
            num_cells (int): The number of cells.
            C (int): The number of channels.

        Returns:
            ModuleList: List of cells.
        """
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
                    mode=self._mode,
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
        r"""Returns a list of alpha parameters.

        Returns:
            ModuleList: List of alpha parameters. The first is used in a normal
                cell and the second is used in the reduction cell.
        """
        shape = (len(darts.CANDIDATES), 1, 1, 1, 1)
        init = ConstantInitializer(0.0)
        n = self._num_choices * (self._num_choices + 3) // 2
        alpha = Mo.ModuleList()
        if self._shared:
            for _ in range(2):
                params = Mo.ParameterList()
                for _ in range(n):
                    params.append(Mo.Parameter(shape, initializer=init))
                alpha.append(params)
        return alpha

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
        if self._shared:
            return self._alpha.get_parameters()
        p = self.get_parameters(grad_only)
        return OrderedDict([(k, v) for k, v in p.items() if 'alpha' in k])

    def summary(self):
        r"""Summary of the model."""
        stats = []
        arch_params = self.get_arch_parameters()
        count = Counter([np.argmax(m.d.flat) for m in arch_params.values()])
        op_names = list(darts.CANDIDATES.keys())
        total = len(arch_params)
        for k in range(len(op_names)):
            name = op_names[k]
            stats.append(name + f' = {count[k]/total*100:.2f}%\t')
        return ''.join(stats)

    def save_parameters(self, path=None, params=None, grad_only=False):
        super().save_parameters(path, params=params, grad_only=grad_only)
        if self._shared:
            # save the architectures
            output_path = os.path.dirname(path)
            save_dart_arch(self, output_path)

    def save_net_nnp(self, path, inp, out, calc_latency=False,
                     func_real_latency=None, func_accum_latency=None,
                     save_params=None):
        super().save_net_nnp(path, inp, out, calc_latency=False,
                             func_real_latency=func_real_latency,
                             func_accum_latency=func_accum_latency,
                             save_params=save_params)
        if self._shared:
            # save the architectures
            save_dart_arch(self, path)

    def visualize(self, path):
        if self._shared:
            # save the architectures
            visualize_dart_arch(path)

    def loss(self, outputs, targets, loss_weights=None):
        loss = F.mean(F.softmax_cross_entropy(outputs[0], targets[0]))
        if len(outputs) == 2:  # use auxiliar head
            loss_weights = loss_weights or (1.0, 1.0)
            aux_loss = F.mean(F.softmax_cross_entropy(outputs[1], targets[0]))
            loss = loss_weights[0] * loss + loss_weights[1] * aux_loss
        return loss


class TrainNet(Model):
    """TrainNet used for DARTS."""

    def __init__(self, in_channels, init_channels, num_cells, num_classes,
                 genotype, num_choices=4, multiplier=4, stem_multiplier=3,
                 drop_path=0, auxiliary=False):
        self._num_ops = len(darts.CANDIDATES)
        self._multiplier = multiplier
        self._init_channels = init_channels
        self._num_cells = num_cells
        self._auxiliary = auxiliary
        self._num_choices = num_choices
        self._drop_path = drop_path

        num_channels = stem_multiplier * init_channels
        genotype_path = os.path.realpath(os.path.join(utils.get_original_cwd(), genotype))
        genotype = json.load(open(genotype_path, 'r'))

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
        return logits if logits_aux is None else (logits, logits_aux)

    def _init_cells(self, num_cells, channel_c, genotype):
        cells = Mo.ModuleList()
        channel_p_p, channel_p, channel_c = channel_c, channel_c, \
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

    def loss(self, outputs, targets, loss_weights=None):
        loss = F.mean(F.softmax_cross_entropy(outputs[0], targets[0]))
        if len(outputs) == 2:  # use auxiliar head
            loss_weights = loss_weights or (1.0, 1.0)
            aux_loss = F.mean(F.softmax_cross_entropy(outputs[1], targets[0]))
            loss = loss_weights[0] * loss + loss_weights[1] * aux_loss
        return loss


class Cell(Mo.Module):
    def __init__(self, channels, reductions, genotype, drop_path):
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
        candidates = list(darts.CANDIDATES.values())

        for i in range(len(cell_arch)):
            for (op_idx, choice_idx) in cell_arch[str(i + 2)]:
                stride = 2 if reductions[-1] and choice_idx < 2 else 1
                self._blocks.append(candidates[op_idx](channels[2], stride))
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
                    choice[-1] = darts.DropPath(self._drop_path)(choice[-1])

            out.append(F.add2(choice[0], choice[1]))

        return F.concatenate(*out[2:], axis=1)
