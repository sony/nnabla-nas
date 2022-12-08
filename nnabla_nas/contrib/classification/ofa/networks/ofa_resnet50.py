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

from collections import OrderedDict
import random

import nnabla as nn
import nnabla.functions as F
import nnabla.logger as logger

from ...base import ClassificationModel
from ..... import module as Mo
from ....common.ofa.layers import ResidualBlock, set_bn_param
from ....common.ofa.utils.common_tools import val2list, make_divisible
from ....common.ofa.utils.common_tools import cross_entropy_loss_with_label_smoothing
from ....common.ofa.utils.common_tools import cross_entropy_loss_with_soft_target
from ....common.ofa.utils.common_tools import init_models
from ....common.ofa.elastic_nn.modules.dynamic_layers import DynamicConvLayer, DynamicLinearLayer
from ....common.ofa.elastic_nn.modules.dynamic_layers import DynamicBottleneckResidualBlock
from ....common.ofa.elastic_nn.modules.dynamic_op import DynamicBatchNorm


class OFAResNet50(ClassificationModel):
    r"""OFAResNet50 Base Class.

    This is the Base Class used for both TrainNet and SearchNet.
    This implementation is based on the PyTorch implementation given in References.

    Args:
        num_classes (int): Number of classes
        bn_param (tuple, optional): BatchNormalization decay rate and eps.
        drop_rate (float, optional): Drop rate used in Dropout in classifier.
            Defaults to 0.1.
        depth_list (int or list of int, optional): Candidates of depth for each layer.
            Defaults to 2.
        expand_ratio_list (float or list of float, optional): Candidates of
            expand ratio for middle bottleneck layers. Defaults to 0.25.
        width_mult_list (float or list of float, optional): Candidates of width
            multiplication ratio for input/output feature size of bottleneck layers.
            Defaults to 1.0.
        weight_init (str, optional): Weight initialization method. Defaults to 'he_fout'.
        weight (str, optional): Path to weight file. Defaults to None.

    References:
        [1] Cai, Han, et al. "Once-for-all: Train one network and specialize it for
            efficient deployment." arXiv preprint arXiv:1908.09791 (2019).
        [2] GitHub implementation of Once-for-All.
            https://github.com/mit-han-lab/once-for-all
    """

    BASE_DEPTH_LIST = [2, 2, 4, 2]
    STAGE_WIDTH_LIST = [256, 512, 1024, 2048]

    def __init__(self,
                 num_classes=1000,
                 bn_param=(0.9, 1e-5),
                 drop_rate=0.1,
                 depth_list=2,
                 expand_ratio_list=0.25,
                 width_mult_list=1.0,
                 weight_init='he_fout',
                 weights=None):

        self._num_classes = num_classes
        self._bn_param = bn_param
        self._drop_rate = drop_rate

        self._depth_list = val2list(depth_list)
        self._expand_ratio_list = val2list(expand_ratio_list)
        self._width_mult_list = val2list(width_mult_list)

        # sort
        self._depth_list.sort()
        self._expand_ratio_list.sort()
        self._width_mult_list.sort()

        input_channel = [
            make_divisible(64 * width_mult) for width_mult in self._width_mult_list
        ]
        mid_input_channel = [
            make_divisible(channel // 2) for channel in input_channel
        ]

        stage_width_list = OFAResNet50.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = [
                make_divisible(width * width_mult) for width_mult in self._width_mult_list
            ]

        n_block_list = [base_depth + max(self._depth_list) for base_depth in OFAResNet50.BASE_DEPTH_LIST]
        stride_list = [1, 2, 2, 2]

        input_stem = [
            DynamicConvLayer(val2list(3), mid_input_channel, (3, 3), stride=(2, 2), use_bn=True),
            ResidualBlock(
                DynamicConvLayer(mid_input_channel, mid_input_channel, (3, 3), stride=(1, 1), use_bn=True),
                Mo.Identity()),
            DynamicConvLayer(mid_input_channel, input_channel, (3, 3), stride=(1, 1), use_bn=True),
        ]
        self.input_stem = Mo.ModuleList(input_stem)
        self.max_pooling = Mo.MaxPool(kernel=(3, 3), stride=(2, 2), pad=(1, 1))

        # blocks
        blocks = []
        for d, width, s in zip(n_block_list, stage_width_list, stride_list):
            for i in range(d):
                stride = (s, s) if i == 0 else (1, 1)
                blocks.append(
                    DynamicBottleneckResidualBlock(
                        input_channel,
                        width,
                        expand_ratio_list=self._expand_ratio_list,
                        kernel=(3, 3),
                        stride=stride,
                        downsample_mode='avgpool_conv')
                )
                input_channel = width
        self.blocks = Mo.ModuleList(blocks)

        # classifier
        self.classifier = Mo.Sequential(
            DynamicLinearLayer(input_channel, num_classes, drop_rate=drop_rate))

        # set bn param (not for dynamic)
        self.set_bn_param(*bn_param)

        # runtime_depth
        self.input_stem_skipping = 0
        self.runtime_depth = [0] * len(n_block_list)
        self.backbone_channel_num = input_channel

        for _, m in self.get_modules():
            if isinstance(m, DynamicBatchNorm):
                if (
                    len(self._expand_ratio_list) > 1
                    or len(self._width_mult_list) > 1
                ):
                    m.use_static_bn = False
                else:
                    m.use_static_bn = True

        # load weights
        if weights is not None:
            self.load_parameters(weights)
        else:
            init_models(self, model_init=weight_init)

    def call(self, input):
        # sample or not
        if self.training:
            self.sample_active_subnet()

        for layer in self.input_stem:
            if (
                self.input_stem_skipping > 0
                and isinstance(layer, ResidualBlock)
                and isinstance(layer.shortcut, Mo.Identity)
            ):
                pass
            else:
                input = layer(input)
        x = self.max_pooling(input)
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                x = self.blocks[idx](x)
        x = F.mean(x, axis=(2, 3), keepdims=True)  # global avg pooling
        x = self.classifier(x)
        return x

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks):
            if not isinstance(block.downsample, Mo.Identity) and len(block_index_list) > 0:
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    def set_valid_arch(self, genotype):
        assert len(genotype) == 3
        depth_list, expand_list, width_list = genotype
        if isinstance(width_list, float):
            width_mult_stages = self._width_mult_list.index(width_list)
        else:
            width_mult_stages = [self._width_mult_list.index(w) for w in width_list]
        self.set_active_subnet(depth_list, expand_list, width_mult_stages)

    def set_active_subnet(self, d=None, e=None, w=None, **kwargs):
        depth = val2list(d, len(self.BASE_DEPTH_LIST) + 1)
        expand_ratio = val2list(e, len(self.blocks))
        width_mult = val2list(w, len(self.BASE_DEPTH_LIST) + 2)

        for block, e in zip(self.blocks, expand_ratio):
            if e is not None:
                block.active_expand_ratio = e

        if width_mult[0] is not None:
            self.input_stem[1].conv.active_out_channel = self.input_stem[0].active_out_channel = \
                self.input_stem[0]._out_channel_list[width_mult[0]]
        if width_mult[1] is not None:
            self.input_stem[2].active_out_channel = self.input_stem[2]._out_channel_list[width_mult[1]]

        if depth[0] is not None:
            self.input_stem_skipping = (depth[0] != max(self._depth_list))
        for stage_id, (block_idx, d, w) in enumerate(zip(self.grouped_block_index, depth[1:], width_mult[2:])):
            if d is not None:
                self.runtime_depth[stage_id] = max(self._depth_list) - d
            if w is not None:
                for idx in block_idx:
                    self.blocks[idx].active_out_channel = self.blocks[idx]._out_channel_list[w]

    def get_random_active_subnet(self):
        # sample expand ratio
        expand_setting = []
        for block in self.blocks:
            expand_setting.append(random.choice(block._expand_ratio_list))

        # sample depth
        depth_setting = [random.choice([max(self._depth_list), min(self._depth_list)])]
        for stage_id in range(len(self.BASE_DEPTH_LIST)):
            depth_setting.append(random.choice(self._depth_list))

        # sample width_mult
        width_mult_setting = [
            random.choice(list(range(len(self.input_stem[0]._out_channel_list)))),
            random.choice(list(range(len(self.input_stem[2]._out_channel_list)))),
        ]
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            stage_first_block = self.blocks[block_idx[0]]
            width_mult_setting.append(
                random.choice(list(range(len(stage_first_block._out_channel_list))))
            )

        arch_config = {
            'd': depth_setting,
            'e': expand_setting,
            'w': width_mult_setting
        }
        return arch_config

    def sample_active_subnet(self):
        arch_config = self.get_random_active_subnet()
        self.set_active_subnet(**arch_config)
        return arch_config

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


class SearchNet(OFAResNet50):
    r"""OFAResNet50 Search Net.

    This defines the search space of OFA-ResNet50 model.

    Args:
        num_classes (int): Number of classes
        bn_param (tuple, optional): BatchNormalization decay rate and eps.
        drop_rate (float, optional): Drop rate used in Dropout in classifier.
            Defaults to 0.1.
        depth_list (int or list of int, optional): Candidates of depth for each layer.
            Defaults to 2.
        expand_ratio_list (float or list of float, optional): Candidates of
            expand ratio for middle bottleneck layers. Defaults to 0.25.
        width_mult_list (float or list of float, optional): Candidates of width
            multiplication ratio for input/output feature size of bottleneck layers.
            Defaults to 1.0.
        weight_init (str, optional): Weight initialization method. Defaults to 'he_fout'.
        weight (str, optional): Path to weight file. Defaults to None.
    """

    def __init__(self,
                 num_classes=1000,
                 bn_param=(0.9, 1e-5),
                 drop_rate=0.1,
                 depth_list=2,
                 expand_ratio_list=0.25,
                 width_mult_list=1.0,
                 weight_init='he_fout',
                 weights=None):
        super(SearchNet, self).__init__(
            num_classes=num_classes, bn_param=bn_param, drop_rate=drop_rate,
            depth_list=depth_list, expand_ratio_list=expand_ratio_list, width_mult_list=width_mult_list,
            weight_init=weight_init, weights=weights)

        if weights is not None:
            self.re_organize_middle_weights()

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        logger.info("Sorting channels according to the importance...")
        for block in self.blocks:
            block.re_organize_middle_weights(expand_ratio_stage)


class TrainNet(SearchNet):
    r"""OFAResNet50 Train Net.

    This builds and initialises the OFA-ResNet50 subnet architecture which
    is passed as a genotype list along with the corresponding depth,
    expand ratio, and width mult candidate list to decode the genotypes.

    Args:
        num_classes (int): Number of classes
        bn_param (tuple, optional): BatchNormalization decay rate and eps.
        drop_rate (float, optional): Drop rate used in Dropout in classifier.
            Defaults to 0.1.
        depth_list (int or list of int, optional): Candidates of depth for each layer.
            Defaults to None.
        expand_ratio_list (float or list of float, optional): Candidates of
            expand ratio for middle bottleneck layers. Defaults to None.
        width_mult_list (float or list of float, optional): Candidates of width
            multiplication ratio for input/output feature size of bottleneck layers.
            Defaults to None.
        genotype (list, optional): A list to operators. Defaults to None.
        weight (str, optional): Path to weight file. Defaults to None.
    """

    def __init__(self, num_classes=1000, bn_param=(0.9, 1e-5), drop_rate=0.1,
                 depth_list=None, expand_ratio_list=None, width_mult_list=None,
                 genotype=None, weights=None):
        if depth_list is None:
            depth_list = [0, 1, 2]
        if expand_ratio_list is None:
            expand_ratio_list = [0.2, 0.25, 0.35]
        if width_mult_list is None:
            width_mult_list = [0.65, 0.8, 1.0]

        super(TrainNet, self).__init__(
            num_classes, bn_param, drop_rate,
            depth_list=depth_list, expand_ratio_list=expand_ratio_list,
            width_mult_list=width_mult_list, weights=weights)

        if genotype is not None:
            assert all(map(genotype['d'].__contains__, ('input_stem1', 'block')))
            assert all(map(genotype['w'].__contains__, ('input_stem1', 'input_stem2', 'block')))
            assert 'block' in genotype['e']
            assert (len(genotype['d']['block']) == 4 and len(genotype['w']['block']) == 4)
            assert len(genotype['e']['block']) == 18

            genotype['d'] = [genotype['d']['input_stem1']] + genotype['d']['block']
            genotype['e'] = genotype['e']['block']
            genotype['w'] = [genotype['w']['input_stem1']] + [genotype['w']['input_stem2']] + genotype['w']['block']
            genotype['w'] = [self._width_mult_list.index(w) for w in genotype['w']]

            self.set_active_subnet(**genotype)
            preserve_weight = True if weights is not None else False

            # input stem
            input_channel = 3
            input_stem = []
            for i, layer in enumerate(self.input_stem):
                if self.input_stem_skipping > 0 \
                        and isinstance(layer, ResidualBlock) and \
                        isinstance(layer.shortcut, Mo.Identity):
                    pass
                else:
                    if isinstance(layer, ResidualBlock):
                        input_stem.append(self.input_stem[i].conv.get_active_subnet(input_channel, preserve_weight))
                        input_channel = input_stem[-1].conv._out_channels
                    else:
                        input_stem.append(self.input_stem[i].get_active_subnet(input_channel, preserve_weight))
                        input_channel = input_stem[-1]._out_channels
            self.input_stem = Mo.ModuleList(input_stem)

            # blocks
            blocks = []
            for stage_id, block_idx in enumerate(self.grouped_block_index):
                depth_param = self.runtime_depth[stage_id]
                active_idx = block_idx[:len(block_idx) - depth_param]
                for idx in active_idx:
                    blocks.append(
                        self.blocks[idx].get_active_subnet(input_channel, preserve_weight))
                    input_channel = blocks[-1].conv3.conv._out_channels
            self.blocks = Mo.ModuleList(blocks)
            self.classifier = self.classifier

    def call(self, x):
        for layer in self.input_stem:
            x = layer(x)
        x = self.max_pooling(x)
        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x)
        x = F.mean(x, axis=(2, 3), keepdims=True)  # global avg pooling
        x = self.classifier(x)
        return x
