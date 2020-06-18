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

import nnabla.functions as F
import numpy as np

from .... import module as Mo
from ..base import ClassificationModel as Model
from ..darts import modules as darts
from ..misc import AuxiliaryHeadCIFAR


class TrainNet(Model):
    r"""TrainNet for ProxylessNAS.

    Args:
        in_channels (int): The number of input channels.
        init_channels (int): The initial number of channels on each cell.
        num_cells (int): The number of cells.
        num_classes (int): The number of classes.
        genotype (str): A file path containing the network weights.
        num_choices (int, optional): The number of choice blocks on each cell.
            Defaults to 4.
        multiplier (int, optional): The multiplier. Defaults to 4.
        stem_multiplier (int, optional): The multiplier used for stem
            convolution. Defaults to 3.
        drop_path (float, optional): Probability of droping paths. Defaults to
            0.
        auxiliary (bool, optional): If uses auxiliary head. Defaults to False.
    """

    def __init__(self, in_channels, init_channels, num_cells, num_classes,
                 genotype, num_choices=4, multiplier=4, stem_multiplier=3,
                 drop_path=0, auxiliary=False):
        self._num_choices = num_choices
        self._multiplier = multiplier
        self._init_channels = init_channels
        self._auxiliary = auxiliary
        self._num_cells = num_cells
        self._drop_path = drop_path

        num_channels = stem_multiplier * init_channels

        # initialize the arch parameters
        self._stem = darts.StemConv(in_channels, num_channels)
        self._cells = self._init_cells(num_cells, num_channels)
        self._ave_pool = Mo.AvgPool(kernel=(8, 8))
        self._linear = Mo.Linear(self._last_channels, num_classes)

        # auxiliary head
        if auxiliary:
            self._auxiliary_head = AuxiliaryHeadCIFAR(
                self._c_auxiliary, num_classes)

        # load weights
        self.load_parameters(genotype)
        for _, module in self.get_modules():
            if isinstance(module, darts.ChoiceBlock):
                idx = np.argmax(module._mixed._alpha.d)
                module._mixed = module._mixed._ops[idx]

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
    def __init__(self, num_choices, multiplier, channels, reductions,
                 drop_path=0.2):
        self._multiplier = multiplier
        self._num_choices = num_choices
        self._channels = channels
        self._drop_path = drop_path

        # preprocess the inputs
        self._prep = Mo.ModuleList()
        if reductions[0]:
            self._prep.append(darts.FactorizedReduce(channels[0], channels[2]))
        else:
            self._prep.append(
                darts.ReLUConvBN(channels[0], channels[2], kernel=(1, 1)))
        self._prep.append(
            darts.ReLUConvBN(channels[1], channels[2], kernel=(1, 1)))

        # build choice blocks
        self._blocks = Mo.ModuleList()
        for i in range(num_choices):
            for j in range(i + 2):
                self._blocks.append(
                    darts.ChoiceBlock(
                        in_channels=channels[2],
                        out_channels=channels[2],
                        is_reduced=j < 2 and reductions[1]
                    )
                )
        self._merge = Mo.Merging(mode='concat', axis=1)

    def call(self, *input):
        """Each cell has two inputs and one output."""
        out = [op(x) for op, x in zip(self._prep, input)]
        offset = 0
        for _ in range(self._num_choices):
            out_temp = []
            for j, h in enumerate(out):
                b = self._blocks[offset + j]
                temp = b(h)
                if self.training and \
                        not isinstance(b._mixed, (Mo.Identity, Mo.Zero)):
                    temp = darts.DropPath(self._drop_path)(temp)
                out_temp.append(temp)
            s = sum(out_temp)
            offset += len(out)
            out.append(s)
        return self._merge(*out[-self._multiplier:])
