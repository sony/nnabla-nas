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

import copy
from collections import OrderedDict

import numpy as np

import nnabla as nn

from .... import module as Mo
from .modules import MBConvLayer, XceptionBlock, set_layer_from_config
from .ofa_utils.common_tools import val2list, make_divisible
from .ofa_modules.static_op import SEModule
from .ofa_modules.dynamic_op import DynamicConv2d, DynamicBatchNorm2d, DynamicSeparableConv2d, DynamicSE
from .modules import build_activation


def adjust_bn_according_to_idx(bn, idx):
    bn._beta.d = np.stack([bn._beta.d[:, i, :, :] for i in idx], axis=1)
    bn._gamma.d = np.stack([bn._gamma.d[:, i, :, :] for i in idx], axis=1)
    bn._mean.d = np.stack([bn._mean.d[:, i, :, :] for i in idx], axis=1)
    bn._var.d = np.stack([bn._var.d[:, i, :, :] for i in idx], axis=1)


def copy_bn(target_bn, src_bn):
    feature_dim = target_bn._n_features
    target_bn._beta.d = src_bn._beta.d[:, :feature_dim, :, :]
    target_bn._gamma.d = src_bn._gamma.d[:, :feature_dim, :, :]
    target_bn._mean.d = src_bn._mean.d[:, :feature_dim, :, :]
    target_bn._var.d = src_bn._var.d[:, :feature_dim, :, :]


class DynamicConvLayer(Mo.Module):

    r"""Convolution-BatchNormalization(optional)-Activation layer
        with dynamic channel selection.

    Args:
        in_channel_list (list of in`): Candidates for the number of active
            input channels.
        out_channel_list (list of int): Candidates for the number of
            output channels.
        kernel (tuple of int): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3, 5). Defaults to (3, 3)
        stride (tuple of int, optional): Stride sizes for dimensions.
            Defaults to (1, 1).
        dilation (tuple of int, optional): Dilation sizes for dimensions.
            Defaults to (1, 1).
        use_bn (bool): If True, BatchNormalization layer is added.
            Defaults to True.
        act_func (str) Type of activation. Defaults to 'relu'.
    """

    def __init__(self, in_channel_list, out_channel_list,
                 kernel=(3, 3), stride=(1, 1), dilation=(1, 1),
                 use_bn=True, act_func='relu6'):
        self._in_channel_list = in_channel_list
        self._out_channel_list = out_channel_list
        self._kernel = kernel
        self._stride = stride
        self._dilation = dilation
        self._use_bn = use_bn
        self._act_func = act_func

        self.conv = DynamicConv2d(
            max_in_channels=max(self._in_channel_list), max_out_channels=max(self._out_channel_list),
            kernel=self._kernel, stride=self._stride, dilation=self._dilation,
        )
        if self._use_bn:
            self.bn = DynamicBatchNorm2d(max(self._out_channel_list), 4)

        self.act = build_activation(self._act_func)

        self.active_out_channel = max(self._out_channel_list)

    def call(self, x):
        self.conv.active_out_channel = self.active_out_channel
        x = self.conv(x)
        if self._use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    def extra_repr(self):
        return (f'in_channel_list={self._in_channel_list}, '
                f'out_channel_list={self._out_channel_list}, '
                f'kernel={self._kernel}, '
                f'stride={self._stride}, '
                f'dilation={self._dilation}, '
                f'use_bn={self._use_bn}, '
                f'act_func={self._act_func}')


class DynamicXPLayer(Mo.Module):

    r"""The Xception block layers with depthwise separable convolution.

    Args:
        in_channel_list (list of int): Candidates for the number of
            active input channels.
        out_channel_list (list of int): Candidates for the number of
            output channels.
        kernel_size_list (list of int or int): Candidates for the kernel size.
            Defaults to 3.
        expand_ratio_list (list of int or int): Candidates for the expand
            ratio. Defaults to 6.
        stride (tuple of int, optional): Stride sizes for dimensions.
            Defaults to (1, 1).
        last_block (bool): Indicates whether the block is last
    """

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=6, stride=(1, 1), depth=3):

        self._in_channel_list = in_channel_list
        self._out_channel_list = out_channel_list
        self._kernel_size_list = val2list(kernel_size_list)
        self._expand_ratio_list = val2list(expand_ratio_list)
        self._stride = stride
        self._runtime_depth = depth
        # build modules
        max_middle_channel = make_divisible(
            round(max(self._in_channel_list) * max(self._expand_ratio_list)))

        self.depth_conv1 = Mo.Sequential(OrderedDict([
            ('act', build_activation('relu')),
            ('dwconv', DynamicSeparableConv2d(max(self._in_channel_list), self._kernel_size_list, self._stride)),
        ]))

        self.point_linear1 = Mo.Sequential(OrderedDict([
            ('ptconv', DynamicConv2d(max(self._in_channel_list), max_middle_channel)),
            ('bn', DynamicBatchNorm2d(max_middle_channel, 4))
        ]))

        self.depth_conv2 = Mo.Sequential(OrderedDict([
            ('act', build_activation('relu')),
            ('dwconv', DynamicSeparableConv2d(max_middle_channel, self._kernel_size_list, self._stride)),
        ]))

        self.point_linear2 = Mo.Sequential(OrderedDict([
            ('ptconv', DynamicConv2d(max_middle_channel, max_middle_channel)),
            ('bn', DynamicBatchNorm2d(max_middle_channel, 4))
        ]))

        self.depth_conv3 = Mo.Sequential(OrderedDict([
            ('act', build_activation('relu')),
            ('dwconv', DynamicSeparableConv2d(max_middle_channel, self._kernel_size_list, self._stride)),
        ]))

        self.point_linear3 = Mo.Sequential(OrderedDict([
            ('ptconv', DynamicConv2d(max_middle_channel, max(self._out_channel_list))),
            ('bn', DynamicBatchNorm2d(max(self._out_channel_list), 4))
        ]))


        self.active_kernel_size = max(self._kernel_size_list)
        self.active_expand_ratio = max(self._expand_ratio_list)
        self.active_out_channel = max(self._out_channel_list)

    def call(self, inp):
        in_channel = inp.shape[1]

        self.depth_conv1.dwconv.active_kernel_size = self.active_kernel_size
        self.point_linear1.ptconv.active_out_channel = \
            make_divisible(round(in_channel * self.active_expand_ratio))

        self.depth_conv2.dwconv.active_kernel_size = self.active_kernel_size
        self.point_linear2.ptconv.active_out_channel = \
            make_divisible(round(in_channel * self.active_expand_ratio))

        self.depth_conv3.dwconv.active_kernel_size = self.active_kernel_size
        self.point_linear3.ptconv.active_out_channel = self.active_out_channel

        # print("#"*30)
        # print(in_channel)
        # print(self.point_linear1.ptconv.active_out_channel)
        # print(make_divisible(round(max(self._in_channel_list) * max(self._expand_ratio_list))))
        # print("#"*30)

        x = self.depth_conv1(inp)
        x = self.point_linear1(x)

        if self._runtime_depth > 1: # runtime depth
            x = self.depth_conv2(x)
            x = self.point_linear2(x)

        if self._runtime_depth > 2: # runtime depth       
            x = self.depth_conv3(x)
            x = self.point_linear3(x)

        # Skip is a simple shortcut ->
        skip = inp
        x += skip
        return x

    def extra_repr(self):
        return (f'in_channel_list={self._in_channel_list}, '
                f'out_channel_list={self._out_channel_list}, '
                f'kernel_size_list={self._kernel_size_list}, '
                f'expand_ratio_list={self._expand_ratio_list}, '
                f'stride={self._stride}, ')

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        importance = np.sum(np.abs(self.point_linear3.ptconv.conv._W.d), axis=(0, 2, 3))
        if expand_ratio_stage > 0:  # ranking channels
            sorted_expand_list = copy.deepcopy(self._expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self._in_channel_list) * expand))
                for expand in sorted_expand_list
            ]
            larger_stage = len(importance)
            base = - len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                smaller_stage = target_width_list[i]
                if larger_stage > smaller_stage:  # not in the original code
                    importance[smaller_stage:larger_stage] += base
                    base += 1e5
                    larger_stage = smaller_stage

        sorted_idx = np.argsort(-importance)
        self.point_linear3.ptconv.conv._W.d = np.stack(
            [self.point_linear3.ptconv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
        adjust_bn_according_to_idx(self.point_linear3.bn.bn, sorted_idx)
        
        self.point_linear2.ptconv.conv._W.d = np.stack(
            [self.point_linear2.ptconv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
        adjust_bn_according_to_idx(self.point_linear2.bn.bn, sorted_idx)
        
        self.point_linear1.ptconv.conv._W.d = np.stack(
            [self.point_linear1.ptconv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
        adjust_bn_according_to_idx(self.point_linear1.bn.bn, sorted_idx)

        self.depth_conv3.dwconv.conv._W.d = np.stack(
            [self.depth_conv3.dwconv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)
        
        self.depth_conv2.dwconv.conv._W.d = np.stack(
            [self.depth_conv2.dwconv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)

        self.depth_conv1.dwconv.conv._W.d = np.stack(
            [self.depth_conv1.dwconv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)

    @property
    def in_channels(self):
        return max(self._in_channel_list)

    @property
    def out_channels(self):
        return max(self._out_channel_list)

    def active_middle_channel(self, in_channel):
        return make_divisible(round(in_channel * self.active_expand_ratio))

    def get_active_subnet(self, in_channel, preserve_weight=True):
        nn.set_auto_forward(True)
        # build the new layer
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        if not preserve_weight:
            return sub_layer

        middle_channel = self.active_middle_channel(in_channel)

        active_filter = self.depth_conv1.dwconv.get_active_filter(in_channel, self.active_kernel_size)
        sub_layer.depth_conv1.dwconv._W.d = active_filter.d

        active_filter = self.depth_conv2.dwconv.get_active_filter(middle_channel, self.active_kernel_size)
        sub_layer.depth_conv2.dwconv._W.d = active_filter.d

        active_filter = self.depth_conv3.dwconv.get_active_filter(middle_channel, self.active_kernel_size)
        sub_layer.depth_conv3.dwconv._W.d = active_filter.d

        copy_bn(sub_layer.point_linear1.bn, self.point_linear1.bn.bn)
        copy_bn(sub_layer.point_linear2.bn, self.point_linear2.bn.bn)
        copy_bn(sub_layer.point_linear3.bn, self.point_linear3.bn.bn)

        sub_layer.point_linear1.ptconv._W.d =\
            self.point_linear1.ptconv.conv._W.d[:middle_channel, :in_channel, :, :]

        sub_layer.point_linear2.ptconv._W.d =\
            self.point_linear2.ptconv.conv._W.d[:middle_channel, :middle_channel, :, :]

        sub_layer.point_linear3.ptconv._W.d =\
            self.point_linear3.ptconv.conv._W.d[:self.active_out_channel, :middle_channel, :, :]

        nn.set_auto_forward(False)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            'name': XceptionBlock.__name__,
            'in_channels': in_channel,
            'out_channels': self.active_out_channel,
            'reps': 3,
            'kernel': (self.active_kernel_size, self.active_kernel_size),
            'stride': self._stride,
            'expand_ratio': self.active_expand_ratio,
            'mid_channels': self.active_middle_channel(in_channel),
        }
