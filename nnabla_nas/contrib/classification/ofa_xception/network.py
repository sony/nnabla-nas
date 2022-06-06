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

import nnabla as nn
import nnabla.functions as F
import random

import nnabla.logger as logger

from ..base import ClassificationModel as Model
from .... import module as Mo
from .modules import ConvLayer, LinearLayer, SeparableConv, XceptionBlock, genotype2subnetlist
from .modules import candidates2subnetlist, genotype2subnetlist, set_bn_param, get_bn_param
from .elastic_modules import DynamicXPLayer
from .ofa_modules.dynamic_op import DynamicBatchNorm2d
from .ofa_utils.common_tools import val2list, make_divisible, cross_entropy_loss_with_label_smoothing


class MyNetwork(Model):
    CHANNEL_DIVISIBLE = 8

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


class SearchNet(MyNetwork):
    r""" Xception41 Search Net
    This implementation is based on the PyTorch implementation.

    Args:
        num_classes (int): Number of classes
        bn_param (tuple, optional): BatchNormalization decay rate and eps.
        drop_rate (float, optional): Drop rate used in Dropout. Defaults to 0.1.
        base_stage_width (list of int, optional): A list of base stage
            channel size. Defaults to None.
        width_mult (float, optional): Multiplier value to base stage channel size.
            Defaults to 1.0.
        op_candidates (str or list of str, optional): Operator choices.
            Defaults to XP6 3x3.
        depth_candidates (int or list of int, optional): Depth choices.
            Defaults to 4.
        weight (str, optional): The path to weight file. Defaults to
            None.

    References:
    [1] Cai, Han, et al. "Once-for-all: Train one network and specialize it for
        efficient deployment." arXiv preprint arXiv:1908.09791 (2019).
    """

    def __init__(self,
                 num_classes=1000,
                 bn_param=(0.9, 1e-5),
                 drop_rate=0.1,
                 base_stage_width=None,
                 op_candidates="XP1 7x7 3",
                 depth_candidates=3,
                 width_mult=1.0,
                 weights=None,
                 output_stride=16):
        self._num_classes = num_classes
        self._bn_param = bn_param
        self._drop_rate = drop_rate
        self._op_candidates = op_candidates
        self._width_mult = width_mult
        self._depth_candidates = depth_candidates
        self._weights = weights
        self._output_stride = output_stride

        op_candidates = val2list(op_candidates, 1)
        ks_list, expand_ratio_list = candidates2subnetlist(op_candidates)
        self._ks_list = val2list(ks_list, 1)
        self._expand_ratio_list = val2list(expand_ratio_list, 1)
        self._depth_list = val2list(depth_candidates)

        # sort
        self._ks_list.sort()
        self._expand_ratio_list.sort()
        self._depth_list.sort()

        base_stage_width = [32, 64, 128, 256, 728, 1024, 1536, 2048]

        expand_1_width = make_divisible(base_stage_width[-3] * self._width_mult)
        expand_2_width = make_divisible(base_stage_width[-2] * self._width_mult)
        last_channel = make_divisible(base_stage_width[-1] * self._width_mult)
        
        first_conv_channel = base_stage_width[0]
        sec_conv_channel = base_stage_width[1]
        mid_block_width = base_stage_width[4]

        n_block_list = [max(self._depth_list)] * 8 # 8 blocks in Xception Middle Flow

        # Entry flow
        # first conv layer
        self.first_conv = ConvLayer(
            3, first_conv_channel, kernel=(3, 3), stride=(2, 2), use_bn=True, act_func='relu', with_bias=False)

        # Second conv layer
        self.second_conv = ConvLayer(first_conv_channel, sec_conv_channel, kernel=(3, 3), stride=(1, 1), dilation=(1, 1),
                                    use_bn=True, act_func=None)

        # entry flow blocks
        self.entryblocks = []
        self.entryblocks.append(XceptionBlock(base_stage_width[1], base_stage_width[2], 2, stride=(2,2), start_with_relu=False))
        self.entryblocks.append(XceptionBlock(base_stage_width[2], base_stage_width[3], 2, stride=(2,2)))
        self.entryblocks.append(XceptionBlock(base_stage_width[3], base_stage_width[4], 2, stride=(2,2)))

        # Middle flow blocks
        self.block_group_info = []
        self.middleblocks = []
        _block_index = 0
        self.block_group_info.append([_block_index+i for i in range(max(self._depth_list))])
        # Here only one set of blocks is needed
        for depth in n_block_list: # 8 blocks with each block having 1,2,3 layers of relu+sep_conv
            self.middleblocks.append(DynamicXPLayer(
                in_channel_list=val2list(mid_block_width), 
                out_channel_list=val2list(mid_block_width), 
                kernel_size_list=self._ks_list,
                expand_ratio_list=self._expand_ratio_list,
                stride=(1,1),
                depth=depth
            ))

        # Exit flow blocks
        self.exitblocks = []
        self.exitblocks.append(XceptionBlock(mid_block_width, expand_1_width, 2, stride=(2,2), grow_first=False))

        self.expand_block1 = SeparableConv(expand_1_width, expand_2_width, (3,3), (1,1), (1,1), use_bn=True, act_fn='relu')
        self.expand_block2 = SeparableConv(expand_2_width, last_channel, (3,3), (1,1), (1,1), use_bn=True, act_fn='relu')

        # use a global average pooling before this FC Layer
        self.classifier = LinearLayer(last_channel, num_classes, drop_rate=drop_rate)

        # set bn param
        self.set_bn_param(decay_rate=bn_param[0], eps=bn_param[1])

        # runtime depth
        self.block_depth_info = [depth for depth in n_block_list]
        self.runtime_depth = [depth for depth in n_block_list]

        # print("#"*30)
        # print(self._expand_ratio_list)
        # print("#"*30)
        if len(self._expand_ratio_list) == 1:
            DynamicBatchNorm2d.GET_STATIC_BN = True
        else:
            DynamicBatchNorm2d.GET_STATIC_BN = False

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
        # blocks
        # just one set of blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            for idx in block_idx:
                depth = self.runtime_depth[idx]
                self.middleblocks[idx]._runtime_depth = depth
                x = self.middleblocks[idx](x)
        x = self.exitblocks[0](x)
        x = self.expand_block1(x)
        x = self.expand_block2(x)
        # Global Avg Pool
        x = F.mean(x, axis=(2, 3), keepdims=True)
        x = F.reshape(x, shape=(x.shape[0], -1))
        return self.classifier(x)

    def set_valid_arch(self, genotype):
        assert(len(genotype) == 8)
        # Here we can assert that genotypes are not skip_connect
        ks_list, expand_ratio_list, depth_list =\
            genotype2subnetlist(self._op_candidates, genotype)
        self.set_active_subnet(ks_list, expand_ratio_list, depth_list)

    @property
    def grouped_block_index(self):
        return self.block_group_info

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
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
                self.runtime_depth[i] = min(self.block_depth_info[i], d)

    def sample_active_subnet(self):
        ks_candidates = self._ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self._expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self._depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.middleblocks))]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.middleblocks))]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting
        }

    def load_parameters(self, path, raise_if_missing=False):
        with nn.parameter_scope('', OrderedDict()):
            nn.load_parameters(path)
            params = nn.get_parameters(grad_only=False)
        self.set_parameters(params, raise_if_missing=raise_if_missing)

    def set_parameters(self, params, raise_if_missing=False):
        for prefix, module in self.get_modules():
            for name, p in module.parameters.items():
                key = prefix + ('/' if prefix else '') + name
                if '/mobile_inverted_conv/' in key:
                    new_key = key.replace('/mobile_inverted_conv/', '/conv/')
                else:
                    new_key = key
                if new_key in params:
                    p.d = params[new_key].d.copy()
                    nn.logger.info(f'`{new_key}` loaded.')
                else:
                    nn.logger.info(f'`{new_key}` does not exist.')
                    if raise_if_missing:
                        raise ValueError(
                            f'A child module {name} cannot be found in '
                            '{this}. This error is raised because '
                            '`raise_if_missing` is specified '
                            'as True. Please turn off if you allow it.')

    def extra_repr(self):
        return (f'num_classes={self._num_classes}, '
                f'drop_rate={self._drop_rate}, '
                f'ks_list={self._ks_list}, '
                f'expand_ratio_list={self._expand_ratio_list}, '
                f'depth_list={self._depth_list}')

    def save_parameters(self, path=None, params=None, grad_only=False):
        super().save_parameters(path, params=params, grad_only=grad_only)


class OFASearchNet(SearchNet):
    def __init__(self,
                 num_classes=1000,
                 bn_param=(0.9, 1e-5),
                 drop_rate=0.1,
                 base_stage_width=None,
                 width_mult=1.0,
                 op_candidates="XP1 7x7 3",
                 depth_candidates=3,
                 weights=None
                 ):
        super(OFASearchNet, self).__init__(
            num_classes=num_classes, bn_param=bn_param, drop_rate=drop_rate,
            base_stage_width=base_stage_width, width_mult=width_mult,
            op_candidates=op_candidates, depth_candidates=depth_candidates, weights=weights)
        if weights is not None:
            self.re_organize_middle_weights()

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        logger.info("Sorting channels according to the importance...")
        for block in self.middleblocks:
            block.re_organize_middle_weights(expand_ratio_stage)


class TrainNet(SearchNet):
    r""" Xception Train Net.
    Args:
        num_classes (int): Number of classes
        bn_param (tuple, optional): BatchNormalization decay rate and eps.
        drop_rate (float, optional): Drop rate used in Dropout. Defaults to 0.1.
        base_stage_width (list of int, optional): A list of base stage
            channel size. Defaults to None.
        width_mult (float, optional): Multiplier value to base stage channel size.
            Defaults to 1.0.
        op_candidates (str or list of str, optional): Operator choices.
            Defaults to XP6 3x3.
        depth_candidates (int or list of int, optional): Depth choices.
            Defaults to 4.
        weight (str, optional): The path to weight file. Defaults to None.
        genotype (list of int, optional): A list to operators, Defaults to None.
    """

    def __init__(self, num_classes=1000, bn_param=(0.9, 1e-5), drop_rate=0.1,
                 base_stage_width=None, width_mult=1,
                 op_candidates=None, depth_candidates=None, genotype=None, weights=None, output_stride=16):

        if op_candidates is None:
            op_candidates = [
                "XP1 3x3 1", "XP1 5x5 1", "XP1 7x7 1",
                "XP1 3x3 2", "XP1 5x5 2", "XP1 7x7 2",
                "XP1 3x3 3", "XP1 5x5 3", "XP1 7x7 3",
            ]
        if depth_candidates is None:
            depth_candidates = [1, 2, 3]

        super(TrainNet, self).__init__(
            num_classes, bn_param, drop_rate, width_mult=width_mult,
            op_candidates=op_candidates, depth_candidates=depth_candidates, weights=weights)

        if genotype is not None:
            assert(len(genotype) == 8)
            ks_list, expand_ratio_list, depth_list = genotype2subnetlist(op_candidates, genotype)
            self.set_active_subnet(ks_list, expand_ratio_list, depth_list)

            preserve_weight = True if weights is not None else False

            blocks = []
            input_channel = self.entryblocks[-1]._out_channels
            for stage_id, block_idx in enumerate(self.block_group_info): # This loop will just run once
                for idx in block_idx:
                    depth = self.runtime_depth[idx]
                    self.middleblocks[idx]._runtime_depth = depth
                    blocks.append(self.middleblocks[idx].get_active_subnet(input_channel, preserve_weight))
                    input_channel = blocks[-1]._out_channels

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
