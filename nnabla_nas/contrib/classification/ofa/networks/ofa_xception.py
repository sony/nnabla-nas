# Copyright (c) 2022 Sony Corporation. All Rights Reserved.
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
import random
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
import nnabla.logger as logger

from ...base import ClassificationModel
from ..... import module as Mo
from ....common.ofa.layers import ConvLayer, LinearLayer, DWSeparableConv, XceptionBlock
from ....common.ofa.layers import set_bn_param, get_bn_param
from ....common.ofa.elastic_nn.modules.dynamic_layers import DynamicMiddleFlowXPBlock
from ....common.ofa.elastic_nn.modules.dynamic_op import DynamicBatchNorm
from ....common.ofa.utils.common_tools import val2list, make_divisible
from ....common.ofa.utils.common_tools import cross_entropy_loss_with_label_smoothing
from ....common.ofa.utils.common_tools import cross_entropy_loss_with_soft_target


def _build_candidates_table():
    kernel_search_space = [3, 5, 7]
    depth_search_space = [1, 2, 3]
    expand_ratio_search_space = [0.6, 0.8, 1]
    candidates_table = {}
    for cur_kernel in kernel_search_space:
        for cur_depth in depth_search_space:
            for cur_expand_ratio in expand_ratio_search_space:
                key = f'XP{cur_expand_ratio} {cur_kernel}x{cur_kernel} {cur_depth}'
                value = {'ks': cur_kernel, 'depth': cur_depth,
                         'expand_ratio': cur_expand_ratio}
                candidates_table[key] = value
    return candidates_table


class ProcessGenotype:

    r""" ProcessGenotype

    This class defines the search space and contains functions
    to process the genotypes and op_candidates to get the subnet
    architecture or the search space.

    Operator candidates: "XP{E} {K}x{K} {D}", E=expand_ratio, K=kernel_size, D=depth_of_block

    Note: If depth of a block==1, expand_ratio will be ignored since we
    just need in_channels and out_channels for a block with a single
    layer. So blocks: ["XP0.6 KxK 1", "XP0.8 KxK 1", "XP1 KxK 1"]
    are equivalent in this architecture design.
    """

    CANDIDATES = _build_candidates_table()

    @classmethod
    def get_search_space(cls, candidates):
        ks_list = []
        expand_list = []
        depth_list = []
        for candidate in candidates:
            ks = cls.CANDIDATES[candidate]['ks']
            e = cls.CANDIDATES[candidate]['expand_ratio']
            depth = cls.CANDIDATES[candidate]['depth']
            if ks not in ks_list:
                ks_list.append(ks)
            if e not in expand_list:
                expand_list.append(e)
            if depth not in depth_list:
                depth_list.append(depth)
        return ks_list, expand_list, depth_list

    @classmethod
    def get_subnet_arch(cls, op_candidates, genotype):
        # We don't need `skip_connect` with the current design of Xception41
        subnet_list = [op_candidates[i] for i in genotype]
        ks_list = [cls.CANDIDATES[subnet]['ks'] for subnet in subnet_list]
        expand_ratio_list = [cls.CANDIDATES[subnet]['expand_ratio'] for subnet in subnet_list]
        depth_list = [cls.CANDIDATES[subnet]['depth'] for subnet in subnet_list]

        assert ([d >= 1 for d in depth_list])
        return ks_list, expand_ratio_list, depth_list


class OFAXceptionNet(ClassificationModel):

    r"""Xception41 Base Class

    This is the Base Class used for both TrainNet and SearchNet.
    This implementation is based on the PyTorch implementation given in References.

    Args:
        num_classes (int): Number of classes
        bn_param (tuple, optional): BatchNormalization decay rate and eps.
        drop_rate (float, optional): Drop rate used in Dropout in classifier.
            Defaults to 0.1.
        base_stage_width (list of int, optional): A list of base stage channel size.
            Defaults to None.
        op_candidates (str or list of str, optional): Operator choices.
            Defaults to "XP1 7x7 3" (the largest block in the search space).
        width_mult (float, optional): Multiplier value to base stage channel size.
            Defaults to 1.0.
        weight (str, optional): The path to weight file. Defaults to
            None.

    References:
        [1] Cai, Han, et al. "Once-for-all: Train one network and specialize it for
            efficient deployment." arXiv preprint arXiv:1908.09791 (2019).
        [2] GitHub implementation of Xception41.
            https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py
    """

    CHANNEL_DIVISIBLE = 8
    NUM_MIDDLE_BLOCKS = 8

    def __init__(self,
                 num_classes=1000,
                 bn_param=(0.9, 1e-5),
                 drop_rate=0.1,
                 base_stage_width=[32, 64, 128, 256, 728, 1024, 1536, 2048],
                 op_candidates="XP1 7x7 3",
                 width_mult=1.0,
                 weights=None):
        self._num_classes = num_classes
        self._bn_param = bn_param
        self._drop_rate = drop_rate
        self._op_candidates = op_candidates
        self._width_mult = width_mult
        self._weights = weights

        op_candidates = val2list(op_candidates, 1)
        ks_list, expand_ratio_list, depth_list = ProcessGenotype.get_search_space(op_candidates)
        self._ks_list = ks_list
        self._expand_ratio_list = expand_ratio_list
        self._depth_list = depth_list

        # sort
        self._ks_list.sort()
        self._expand_ratio_list.sort()
        self._depth_list.sort()

        # width_mult scaled block widths
        width_list = []
        for base_width in base_stage_width:
            width = make_divisible(base_width * self._width_mult, OFAXceptionNet.CHANNEL_DIVISIBLE)
            width_list.append(width)

        # width_mult scaled middle and exit block widths
        mid_block_width = width_list[-4]
        last_channel = width_list[-1]

        # list of max supported depth for each block in the middle flow
        self._middle_flow_max_depth_list = [max(self._depth_list)] * OFAXceptionNet.NUM_MIDDLE_BLOCKS

        # Entry flow
        # first conv layer
        self.first_conv = ConvLayer(3, width_list[0],
                                    kernel=(3, 3), stride=(2, 2), use_bn=True, act_func='relu', with_bias=False)

        # Second conv layer
        self.second_conv = ConvLayer(width_list[0], width_list[1],
                                     kernel=(3, 3), stride=(1, 1), use_bn=True, act_func='relu', with_bias=False)

        # entry flow blocks
        self.entryblocks = []
        self.entryblocks.append(XceptionBlock(
            width_list[1], width_list[2], 2, stride=(2, 2), start_with_relu=False))
        self.entryblocks.append(XceptionBlock(
            width_list[2], width_list[3], 2, stride=(2, 2)))
        self.entryblocks.append(XceptionBlock(
            width_list[3], mid_block_width, 2, stride=(2, 2)))

        self.entryblocks = Mo.ModuleList(self.entryblocks)

        # Middle flow blocks
        self.middleblocks = []
        for depth in self._middle_flow_max_depth_list:
            # 8 blocks with each block having 1/2/3 layers of relu+sep_conv
            self.middleblocks.append(DynamicMiddleFlowXPBlock(
                in_channel_list=val2list(mid_block_width),
                out_channel_list=val2list(mid_block_width),
                kernel_size_list=self._ks_list,
                expand_ratio_list=self._expand_ratio_list,
                stride=(1, 1),
                depth=depth
            ))
        self.middleblocks = Mo.ModuleList(self.middleblocks)

        # Exit flow blocks
        self.exitblocks = []
        self.exitblocks.append(XceptionBlock(mid_block_width, width_list[-3], 2, stride=(2, 2), grow_first=False))

        self.exitblocks = Mo.ModuleList(self.exitblocks)

        self.expand_block1 = DWSeparableConv(width_list[-3], width_list[-2], (3, 3),
                                             (1, 1), (1, 1), use_bn=True, act_fn='relu')
        self.expand_block2 = DWSeparableConv(width_list[-2], last_channel, (3, 3),
                                             (1, 1), (1, 1), use_bn=True, act_fn='relu')

        # use a global average pooling before this FC Layer
        self.classifier = LinearLayer(last_channel, num_classes, drop_rate=drop_rate)

        # set bn param
        self.set_bn_param(decay_rate=bn_param[0], eps=bn_param[1])

        # initialise the runtime depth of each block in the middle flow
        self.middle_flow_runtime_depth_list = self._middle_flow_max_depth_list.copy()

        # set static/dynamic bn
        for _, m in self.get_modules():
            if isinstance(m, DynamicBatchNorm):
                if len(self._expand_ratio_list) > 1:
                    m.use_static_bn = False
                else:
                    m.use_static_bn = True

        if weights is not None:
            self.load_parameters(weights)

    def call(self, x):
        # sample or not
        if self.training:
            self.sample_active_subnet()

        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.entryblocks[0](x)
        x = self.entryblocks[1](x)
        x = self.entryblocks[2](x)
        # xception has only one stage in the middle flow
        for middleblock, runtime_depth in zip(self.middleblocks, self.middle_flow_runtime_depth_list):
            middleblock.runtime_depth = runtime_depth
            x = middleblock(x)
        x = self.exitblocks[0](x)
        x = self.expand_block1(x)
        x = self.expand_block2(x)
        # Global Avg Pool
        x = F.mean(x, axis=(2, 3), keepdims=True)
        x = F.reshape(x, shape=(x.shape[0], -1))
        return self.classifier(x)

    def set_valid_arch(self, genotype):
        assert (len(genotype) == OFAXceptionNet.NUM_MIDDLE_BLOCKS)
        ks_list, expand_ratio_list, depth_list =\
            ProcessGenotype.get_subnet_arch(self._op_candidates, genotype)
        self.set_active_subnet(ks_list, expand_ratio_list, depth_list)

    def set_active_subnet(self, ks, e, d, **kwargs):
        ks = val2list(ks)
        expand_ratio = val2list(e)
        depth = val2list(d)

        for block, k, e in zip(self.middleblocks, ks, expand_ratio):
            if k is not None:
                block.active_kernel_size = k
            if e is not None:
                block.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.middle_flow_runtime_depth_list[i] = min(self._middle_flow_max_depth_list[i], d)

    def sample_active_subnet(self):
        ks_candidates = self._ks_list
        expand_candidates = self._expand_ratio_list
        depth_candidates = self._depth_list

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(OFAXceptionNet.NUM_MIDDLE_BLOCKS)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(OFAXceptionNet.NUM_MIDDLE_BLOCKS)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(OFAXceptionNet.NUM_MIDDLE_BLOCKS)]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting
        }

    def set_bn_param(self, decay_rate, eps, **kwargs):
        r"""Sets decay_rate and eps to batchnormalization layers.

        Args:
            decay_rate (float): Deccay rate of running mean and variance.
            eps (float):Tiny value to avoid zero division by std.
        """
        set_bn_param(self, decay_rate, eps, **kwargs)

    def get_bn_param(self):
        r"""Return dict of batchnormalization params.

        Returns:
            dict: A dictionary containing decay_rate and eps of batchnormalization
        """
        return get_bn_param(self)

    def loss(self, outputs, targets, loss_weights=None):
        r"""Return loss computed from a list of outputs and list of targets.

        Args:
            outputs (list of nn.Variable):
                A list of output variables computed from the model.
            targets (list of nn.Variable):
                A list of target variables loaded from the data.
            loss_weights (list of float, optional):
                A list specifying scalar coefficients to weight the loss
                contributions of different model outputs.
                It is expected to have a 1:1 mapping to model outputs.
                Defaults to None.
        Returns:
            nn.Variable: A scalar NNabla Variable represents the loss.
        """
        return cross_entropy_loss_with_label_smoothing(outputs[0], targets[0])

    def kd_loss(self, outputs, logits, targets, loss_weights=None):
        soft_label = F.softmax(logits[0], axis=1)
        soft_label.apply(persistent=True)
        kd_loss = cross_entropy_loss_with_soft_target(outputs[0], soft_label)
        return kd_loss

    def get_net_parameters(self, grad_only=False):
        r"""Returns an `OrderedDict` containing architecture parameters.

        Args:
            grad_only (bool, optional): If sets to `True`, then only parameters
                with `need_grad=True` are returned. Defaults to False.

        Returns:
            OrderedDict: A dictionary containing parameters.
        """
        p = self.get_parameters(grad_only)
        return OrderedDict([(k, v) for k, v in p.items()])

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

    def set_parameters(self, params, raise_if_missing=False):
        for prefix, module in self.get_modules():
            for name, p in module.parameters.items():
                key = prefix + ('/' if prefix else '') + name
                if key in params and p.shape == params[key].shape:
                    p.d = params[key].d.copy()
                    nn.logger.info(f'`{key}` loaded.')
                else:
                    nn.logger.info(f'`{key}` does not exist.')
                    if raise_if_missing:
                        raise ValueError(
                            f'A child module {name} cannot be found in '
                            '{this}. This error is raised because '
                            '`raise_if_missing` is specified '
                            'as True. Please turn off if you allow it.')

    def extra_repr(self):
        repr = ""
        for var in vars(self):
            var_value = getattr(self, var)
            repr += f'{var}='
            repr += f'{var_value}, '

        repr += ')'
        return repr


class SearchNet(OFAXceptionNet):

    r"""Xception41 Search Net.

    This defines the search space of OFA-Xception Model.

    Args:
        num_classes (int): Number of classes
        bn_param (tuple, optional): BatchNormalization decay rate and eps.
        drop_rate (float, optional): Drop rate used in Dropout of classifier.
            Defaults to 0.1.
        base_stage_width (list of int, optional): A list of base stage channel size.
            Defaults to [32, 64, 128, 256, 728, 1024, 1536, 2048].
        width_mult (float, optional): Multiplier value to base stage channel size.
            Defaults to 1.0.
        op_candidates (str or list of str, optional): Operator choices.
            Defaults to "XP1 7x7 3" (the largest block in the search space)
        weights (str, optional): The path to weight file. Defaults to None.
    """

    def __init__(self,
                 num_classes=1000,
                 bn_param=(0.9, 1e-5),
                 drop_rate=0.1,
                 base_stage_width=[32, 64, 128, 256, 728, 1024, 1536, 2048],
                 width_mult=1.0,
                 op_candidates="XP1 7x7 3",
                 weights=None
                 ):
        super(SearchNet, self).__init__(
            num_classes=num_classes, bn_param=bn_param, drop_rate=drop_rate,
            base_stage_width=base_stage_width, width_mult=width_mult,
            op_candidates=op_candidates, weights=weights)
        if weights is not None:
            self.re_organize_middle_weights()

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        logger.info("Sorting channels according to the importance...")
        for block in self.middleblocks:
            block.re_organize_middle_weights(expand_ratio_stage)


class TrainNet(OFAXceptionNet):
    r"""Xception41 Train Net.

    This builds and initialises the OFA-Xception subnet architecture which
    is passed as a genotype list along with the corresponding op_candidates
    list to decode the genotypes.

    Args:
        num_classes (int): Number of classes
        bn_param (tuple, optional): BatchNormalization decay rate and eps.
        drop_rate (float, optional): Drop rate used in Dropout of classifier.
            Defaults to 0.1.
        base_stage_width (list of int, optional): A list of base stage channel size.
            Defaults to [32, 64, 128, 256, 728, 1024, 1536, 2048].
        width_mult (float, optional): Multiplier value to base stage channel size.
            Defaults to 1.0.
        op_candidates (str or list of str, optional): Operator choices.
            Defaults to None. [Necessary Argument]
        genotype (list of int, optional): A list to operators. Defaults to None.
        weights (str, optional): The path to weight file. Defaults to None.
    """

    def __init__(self,
                 num_classes=1000,
                 bn_param=(0.9, 1e-5),
                 drop_rate=0.1,
                 base_stage_width=[32, 64, 128, 256, 728, 1024, 1536, 2048],
                 width_mult=1,
                 op_candidates=None,
                 genotype=None,
                 weights=None):

        if op_candidates is None:
            op_candidates = [
                "XP1 3x3 1", "XP1 3x3 2", "XP1 3x3 3",
                "XP0.8 3x3 1", "XP0.8 3x3 2", "XP0.8 3x3 3",
                "XP0.6 3x3 1", "XP0.6 3x3 2", "XP0.6 3x3 3",
                "XP1 5x5 1", "XP1 5x5 2", "XP1 5x5 3",
                "XP0.8 5x5 1", "XP0.8 5x5 2", "XP0.8 5x5 3",
                "XP0.6 5x5 1", "XP0.6 5x5 2", "XP0.6 5x5 3",
                "XP1 7x7 1", "XP1 7x7 2", "XP1 7x7 3",
                "XP0.8 7x7 1", "XP0.8 7x7 2", "XP0.8 7x7 3",
                "XP0.6 7x7 1", "XP0.6 7x7 2", "XP0.6 7x7 3"
            ]

        super(TrainNet, self).__init__(
            num_classes, bn_param, drop_rate, width_mult=width_mult,
            base_stage_width=base_stage_width, op_candidates=op_candidates, weights=weights)

        if genotype is not None:
            assert (len(genotype) == OFAXceptionNet.NUM_MIDDLE_BLOCKS)
            ks_list, expand_ratio_list, depth_list = ProcessGenotype.get_subnet_arch(op_candidates, genotype)
            self.set_active_subnet(ks_list, expand_ratio_list, depth_list)

            preserve_weight = True if weights is not None else False

            blocks = []
            input_channel = self.entryblocks[-1].out_channels
            for middleblock, runtime_depth in zip(self.middleblocks, self.middle_flow_runtime_depth_list):
                middleblock.runtime_depth = runtime_depth
                blocks.append(middleblock.get_active_subnet(input_channel, preserve_weight))
                input_channel = blocks[-1].out_channels

            self.middleblocks = Mo.ModuleList(blocks)

    def call(self, x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.entryblocks[0](x)
        x = self.entryblocks[1](x)
        x = self.entryblocks[2](x)
        for idx in range(len(self.middleblocks)):
            x = self.middleblocks[idx](x)
        x = self.exitblocks[0](x)
        x = self.expand_block1(x)
        x = self.expand_block2(x)
        x = F.mean(x, axis=(2, 3), keepdims=True)
        x = F.reshape(x, shape=(x.shape[0], -1))
        return self.classifier(x)
