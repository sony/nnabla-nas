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

from collections import OrderedDict
import random

import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.logger as logger

from ...base import ClassificationModel
from ..... import module as Mo
from ....common.ofa.layers import ResidualBlock, ConvLayer, LinearLayer, MBConvLayer, set_bn_param
from ....common.ofa.utils.common_tools import val2list, make_divisible
from ....common.ofa.utils.common_tools import cross_entropy_loss_with_label_smoothing
from ....common.ofa.utils.common_tools import cross_entropy_loss_with_soft_target
from ....common.ofa.utils.common_tools import init_models
from ....common.ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer
from ....common.ofa.elastic_nn.modules.dynamic_op import DynamicBatchNorm


CANDIDATES = {
    'MB3 3x3': {'ks': 3, 'expand_ratio': 3},
    'MB3 5x5': {'ks': 5, 'expand_ratio': 3},
    'MB3 7x7': {'ks': 7, 'expand_ratio': 3},
    'MB4 3x3': {'ks': 3, 'expand_ratio': 4},
    'MB4 5x5': {'ks': 5, 'expand_ratio': 4},
    'MB4 7x7': {'ks': 7, 'expand_ratio': 4},
    'MB6 3x3': {'ks': 3, 'expand_ratio': 6},
    'MB6 5x5': {'ks': 5, 'expand_ratio': 6},
    'MB6 7x7': {'ks': 7, 'expand_ratio': 6},
    'skip_connect': {'ks': None, 'expand_ratio': None},
}


def candidates2subnetlist(candidates):
    ks_list = []
    expand_list = []
    for candidate in candidates:
        ks = CANDIDATES[candidate]['ks']
        e = CANDIDATES[candidate]['expand_ratio']
        if ks not in ks_list:
            ks_list.append(ks)
        if e not in expand_list:
            expand_list.append(e)
    return ks_list, expand_list


def genotype2subnetlist(op_candidates, genotype):
    op_candidates.append('skip_connect')
    subnet_list = [op_candidates[i] for i in genotype]
    ks_list = [CANDIDATES[subnet]['ks'] if subnet != 'skip_connect'
               else 3 for subnet in subnet_list]
    expand_ratio_list = [CANDIDATES[subnet]['expand_ratio'] if subnet != 'skip_connect'
                         else 4 for subnet in subnet_list]
    depth_list = []
    d = 0
    for i, subnet in enumerate(subnet_list):
        if subnet == 'skip_connect':
            if d > 1:
                depth_list.append(d)
                d = 0
        elif d == 4:
            depth_list.append(d)
            d = 1
        elif i == len(subnet_list) - 1:
            depth_list.append(d + 1)
        else:
            d += 1
    assert ([d > 1 for d in depth_list])
    return ks_list, expand_ratio_list, depth_list


class OFAMbv3Net(ClassificationModel):
    r"""MobileNet V3 Search Net.

    Args:
        num_classes (int, optional): Number of classes. Defaults to 1000.
        bn_param (tuple, optional): BatchNormalization decay rate and eps. Defaults to (0.9, 1e-5).
        drop_rate (float, optional): Drop rate used in Dropout. Defaults to 0.1.
        base_stage_width (list of int, optional): A list of base stage channel size. Defaults to None.
        width_mult (float, optional): Multiplier value to base stage channel size. Defaults to 1.0.
        op_candidates (str or list of str, optional): Operator choices. Defaults to "MB6 3x3".
        depth_candidates (int or list of int, optional): Depth choices. Defaults to 4.
        compound (bool, optional): Use CompOFA or not. Defaults to False.
        fixed_kernel (bool, optional): Fix kernel or not. Defaults to False.
        weight_init (str, optional): Weight initializer. Defaults to 'he_fout'.
        weights (str, optional): The relative path to weight file. Defaults to None.

    References:
        [1] Cai, Han, et al. "Once-for-all: Train one network and specialize it for
            efficient deployment." arXiv preprint arXiv:1908.09791 (2019).
    """
    CHANNEL_DIVISIBLE = 8

    def __init__(self,
                 num_classes=1000,
                 bn_param=(0.9, 1e-5),
                 drop_rate=0.1,
                 base_stage_width=None,
                 width_mult=1.0,
                 op_candidates="MB6 3x3",
                 depth_candidates=4,
                 compound=False,
                 fixed_kernel=False,
                 weight_init='he_fout',
                 weights=None):

        self._num_classes = num_classes
        self._bn_param = bn_param
        self._drop_rate = drop_rate
        self._width_mult = width_mult
        self._op_candidates = op_candidates
        self._depth_candidates = depth_candidates
        self._weights = weights

        op_candidates = val2list(op_candidates, 1)
        ks_list, expand_ratio_list = candidates2subnetlist(op_candidates)
        self._ks_list = val2list(ks_list, 1)
        self._expand_ratio_list = val2list(expand_ratio_list, 1)
        self._depth_list = val2list(depth_candidates)

        # compofa
        self._compound = compound
        self._fixed_kernel = fixed_kernel

        # sort
        self._ks_list.sort()
        self._expand_ratio_list.sort()
        self._depth_list.sort()

        base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

        final_expand_width = make_divisible(
            base_stage_width[-2] * self._width_mult, OFAMbv3Net.CHANNEL_DIVISIBLE)
        last_channel = make_divisible(base_stage_width[-1] * self._width_mult)

        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
        se_stages = [False, False, True, False, True, True]
        n_block_list = [1] + [max(self._depth_list)] * 5
        width_list = []
        for base_width in base_stage_width[:-2]:
            width = make_divisible(base_width * self._width_mult, OFAMbv3Net.CHANNEL_DIVISIBLE)
            width_list.append(width)

        input_channel, first_block_dim = width_list[0], width_list[1]
        # first conv layer
        self.first_conv = ConvLayer(
            3, input_channel, kernel=(3, 3), stride=(2, 2), act_func='h_swish')
        first_block_conv = MBConvLayer(
            input_channel, first_block_dim, kernel=(3, 3), stride=(stride_stages[0], stride_stages[0]),
            expand_ratio=1, act_func=act_stages[0], use_se=se_stages[0],
        )
        first_block = ResidualBlock(
            first_block_conv,
            Mo.Identity()
        )

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1
        feature_dim = first_block_dim
        for width, n_block, s, act_func, use_se in zip(width_list[2:], n_block_list[1:],
                                                       stride_stages[1:], act_stages[1:], se_stages[1:]):
            self.block_group_info.append(
                [_block_index + i for i in range(n_block)])
            _block_index += n_block
            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = (s, s)
                else:
                    stride = (1, 1)
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim),
                    out_channel_list=val2list(output_channel),
                    kernel_size_list=self._ks_list, expand_ratio_list=self._expand_ratio_list,
                    stride=stride, act_func=act_func, use_se=use_se,
                )
                if stride == (1, 1) and feature_dim == output_channel:
                    shortcut = Mo.Identity()
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel

        self.blocks = Mo.ModuleList(blocks)
        # final expand layer, feature mix layer & classifier
        self.final_expand_layer = ConvLayer(
            feature_dim, final_expand_width, kernel=(1, 1), act_func='h_swish'
        )
        self.feature_mix_layer = ConvLayer(
            final_expand_width, last_channel, kernel=(1, 1), with_bias=False, use_bn=False, act_func='h_swish'
        )
        self.classifier = LinearLayer(
            last_channel, num_classes, drop_rate=drop_rate)

        # set bn param
        self.set_bn_param(decay_rate=bn_param[0], eps=bn_param[1])

        # runtime depth
        self.runtime_depth = [len(block_idx)
                              for block_idx in self.block_group_info]
        self.backbone_channel_num = final_expand_width

        # set static/dynamic bn
        for _, m in self.get_modules():
            if isinstance(m, DynamicBatchNorm):
                if len(self._expand_ratio_list) > 1:
                    m.use_static_bn = False
                else:
                    m.use_static_bn = True

        if weights is not None:
            self.load_parameters(weights)
        else:
            init_models(self, model_init=weight_init)

    def call(self, x):
        # sample or not
        if self.training:
            self.sample_active_subnet()

        x = self.first_conv(x)
        x = self.blocks[0](x)
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)
        x = self.final_expand_layer(x)
        x = F.mean(x, axis=(2, 3), keepdims=True)  # global avg pooling
        x = self.feature_mix_layer(x)
        x = F.reshape(x, shape=(x.shape[0], -1))
        return self.classifier(x)

    def set_valid_arch(self, genotype):
        assert (len(genotype) == 20)
        ks_list, expand_ratio_list, depth_list =\
            genotype2subnetlist(self._op_candidates, genotype)
        self.set_active_subnet(ks_list, expand_ratio_list, depth_list)

    @property
    def grouped_block_index(self):
        return self.block_group_info

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
        if self._fixed_kernel:
            assert ks is None, "You tried to set kernel size for a fixed kernel network!"
            ks = []
            kernel_stages = [3, 3, 5, 3, 3, 5]
            for k in kernel_stages[1:]:
                ks.extend([k] * 4)

        ks = val2list(ks, len(self.blocks) - 1)
        expand_ratio = val2list(e, len(self.blocks) - 1)
        depth = val2list(d, len(self.block_group_info))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def sample_active_subnet(self):
        if self._compound:
            return self.sample_compound_subnet()

        ks_candidates = self._ks_list
        expand_candidates = self._expand_ratio_list
        depth_candidates = self._depth_list

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [
                ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [
                expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(
                len(self.block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting
        }

    def sample_compound_subnet(self):

        def clip_expands(expands):
            low = min(self._expand_ratio_list)
            expands = list(set(np.clip(expands, low, None)))
            return expands

        ks_candidates = self._ks_list
        depth_candidates = self._depth_list

        mapping = {
            2: clip_expands([3, ]),
            3: clip_expands([4, ]),
            4: clip_expands([6, ]),
        }

        # used in in case of unbalanced distribution to sample proportional w/ cardinality
        combinations_per_depth = {
            d: len(mapping[d])**d for d in depth_candidates}
        sum_combinations = sum(combinations_per_depth.values())
        depth_sampling_weights = {
            k: v / sum_combinations for k, v in combinations_per_depth.items()}

        depth_setting = []
        expand_setting = []
        for block_idx in self.block_group_info:
            # for each block, sample a random depth weighted by the number of combinations
            # for each layer in block, sample from corresponding expand ratio
            sampled_d = np.random.choice(
                depth_candidates, p=list(depth_sampling_weights.values()))
            corresp_e = mapping[sampled_d]

            depth_setting.append(sampled_d)
            for _ in range(len(block_idx)):
                expand_setting.append(random.choice(corresp_e))

        if self._fixed_kernel:
            ks_setting = None
        else:
            # sample kernel size
            ks_setting = []
            if not isinstance(ks_candidates[0], list):
                ks_candidates = [
                    ks_candidates for _ in range(len(self.blocks) - 1)]
            for k_set in ks_candidates:
                k = random.choice(k_set)
                ks_setting.append(k)
        self.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    def extra_repr(self):
        repr = ""
        for var in vars(self):
            var_value = getattr(self, var)
            repr += f'{var}='
            repr += f'{var_value}, '

        repr += ')'
        return repr

    def set_bn_param(self, decay_rate, eps, **kwargs):
        r"""Sets decay_rate and eps to batchnormalization layers.

        Args:
            decay_rate (float): Deccay rate of running mean and variance.
            eps (float):Tiny value to avoid zero division by std.
        """
        set_bn_param(self, decay_rate, eps, **kwargs)

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


class SearchNet(OFAMbv3Net):
    def __init__(self,
                 num_classes=1000,
                 bn_param=(0.9, 1e-5),
                 drop_rate=0.1,
                 base_stage_width=None,
                 width_mult=1.0,
                 op_candidates="MB6 3x3",
                 depth_candidates=4,
                 compound=False,
                 fixed_kernel=False,
                 weight_init="he_fout",
                 weights=None
                 ):
        super(SearchNet, self).__init__(
            num_classes=num_classes, bn_param=bn_param, drop_rate=drop_rate,
            base_stage_width=base_stage_width, width_mult=width_mult,
            op_candidates=op_candidates, depth_candidates=depth_candidates,
            compound=compound, fixed_kernel=fixed_kernel,
            weight_init=weight_init, weights=weights)

        if weights is not None:
            self.re_organize_middle_weights()

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        logger.info("Sorting channels according to the importance...")
        for block in self.blocks[1:]:
            block.conv.re_organize_middle_weights(expand_ratio_stage)


class TrainNet(OFAMbv3Net):
    r"""MobileNet V3 Train Net.

    Args:
        num_classes (int, optional): Number of classes. Defaults to 1000.
        bn_param (tuple, optional): BatchNormalization decay rate and eps. Defaults to (0.9, 1e-5).
        drop_rate (float, optional): Drop rate used in Dropout. Defaults to 0.1.
        base_stage_width (list of int, optional): A list of base stage channel size. Defaults to None.
        width_mult (float, optional): Multiplier value to base stage channel size. Defaults to 1.0.
        op_candidates (str or list of str, optional): Operator choices. Defaults to None.
        depth_candidates (int or list of int, optional): Depth choices. Defaults to None.
        genotype (list of int, optional): A list to operators. Defaults to None.
        weights (str, optional): Relative path to the weights file. Defaults to None.
    """

    def __init__(self, num_classes=1000, bn_param=(0.9, 1e-5), drop_rate=0.1,
                 base_stage_width=None, width_mult=1,
                 op_candidates=None, depth_candidates=None, genotype=None, weights=None):

        if op_candidates is None:
            op_candidates = [
                "MB3 3x3", "MB3 5x5", "MB3 7x7",
                "MB4 3x3", "MB4 5x5", "MB4 7x7",
                "MB6 3x3", "MB6 5x5", "MB6 7x7",
            ]
        if depth_candidates is None:
            depth_candidates = [2, 3, 4]

        super(TrainNet, self).__init__(
            num_classes, bn_param, drop_rate, width_mult=width_mult,
            op_candidates=op_candidates, depth_candidates=depth_candidates, weights=weights)

        if genotype is not None:
            assert (len(genotype) == 20)
            ks_list, expand_ratio_list, depth_list = genotype2subnetlist(
                op_candidates, genotype)
            self.set_active_subnet(ks_list, expand_ratio_list, depth_list)

            preserve_weight = True if weights is not None else False

            blocks = [self.blocks[0]]
            input_channel = blocks[0].conv._out_channels
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                stage_blocks = []
                for idx in active_idx:
                    stage_blocks.append(ResidualBlock(
                        self.blocks[idx].conv.get_active_subnet(
                            input_channel, preserve_weight),
                        self.blocks[idx].shortcut
                    ))
                    input_channel = stage_blocks[-1].conv._out_channels
                blocks += stage_blocks

            self.blocks = Mo.ModuleList(blocks)
            self.final_expand_layer = self.final_expand_layer
            self.feature_mix_layer = self.feature_mix_layer
            self.classifier = self.classifier

    def call(self, x):
        x = self.first_conv(x)
        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x)
        x = self.final_expand_layer(x)
        x = F.mean(x, axis=(2, 3), keepdims=True)
        x = self.feature_mix_layer(x)
        x = F.reshape(x, shape=(x.shape[0], -1))
        x = self.classifier(x)
        return x
