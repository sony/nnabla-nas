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

from ...... import module as Mo
from ...layers import ConvLayer, LinearLayer, MBConvLayer, SEModule, XceptionBlock, BottleneckResidualBlock
from ...layers import set_layer_from_config, build_activation, get_extra_repr
from ...utils.common_tools import val2list, make_divisible
from .dynamic_op import DynamicConv, DynamicLinear, DynamicBatchNorm, DynamicDepthwiseConv, DynamicSE


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

        self.conv = DynamicConv(
            max_in_channels=max(self._in_channel_list), max_out_channels=max(self._out_channel_list),
            kernel=self._kernel, stride=self._stride, dilation=self._dilation,
        )
        if self._use_bn:
            self.bn = DynamicBatchNorm(max(self._out_channel_list), 4)

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
        return get_extra_repr(self)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        with nn.auto_forward():
            sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))

            if not preserve_weight:
                return sub_layer

            sub_layer.conv._W.d = self.conv.get_active_filter(
                self.active_out_channel, in_channel).d
            if self._use_bn:
                copy_bn(sub_layer.bn, self.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            "name": ConvLayer.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel": self._kernel,
            "stride": self._stride,
            "dilation": self._dilation,
            "use_bn": self._use_bn,
            "act_func": self._act_func,
        }


class DynamicLinearLayer(Mo.Module):

    r"""Dynamic version of affine or fully connected layer with dropout.

        Args:
            in_features_list (list of int): Candidates of each input sample.
            in_features_list (list of int): Candidates of each output sample.
            bias (bool, optional): Specify whether to include the bias term.
                Defaults to True.
            drop_rate (float, optional): Dropout ratio applied to parameters.
                Defaults to 0.
    """

    def __init__(self, in_features_list, out_features, bias=True, drop_rate=0):

        self._in_features_list = in_features_list
        self._out_features = out_features
        self._bias = bias
        self._drop_rate = drop_rate

        if self._drop_rate > 0:
            self.dropout = Mo.Dropout(self._drop_rate)
        else:
            self.dropout = None
        self.linear = DynamicLinear(
            max(self._in_features_list), self._out_features, bias=self._bias)

    def call(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)

    def extra_repr(self):
        return get_extra_repr(self)

    def get_active_subnet(self, in_features, preserve_weight=True):
        with nn.auto_forward():
            sub_layer = LinearLayer(
                in_features, self._out_features, self._bias, drop_rate=self._drop_rate)
            if not preserve_weight:
                return sub_layer

            sub_layer.linear._W = self.linear.get_active_weight(self._out_features, in_features).d
            if self._bias:
                sub_layer.linear._b.d = self.linear.get_active_bias(self._out_features).d

        return sub_layer

    def get_active_subnet_config(self, in_features):
        return {
            "name": LinearLayer.__name__,
            "in_features": in_features,
            "out_features": self._out_features,
            "bias": self._bias,
            "dropout_rate": self._drop_rate,
        }


class DynamicMBConvLayer(Mo.Module):

    r"""The inverted layer with optional squeeze-and-expand.

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
        act_func (str) Type of activation. Defaults to 'relu'.
        use_se (bool, optional): If True, squeeze-and-expand module
            is used. Defaults to False.
    """

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=6, stride=(1, 1),
                 act_func='relu6', use_se=False):

        self._in_channel_list = in_channel_list
        self._out_channel_list = out_channel_list
        self._kernel_size_list = val2list(kernel_size_list)
        self._expand_ratio_list = val2list(expand_ratio_list)
        self._stride = stride
        self._act_func = act_func
        self._use_se = use_se

        # build modules
        max_middle_channel = make_divisible(
            round(max(self._in_channel_list) * max(self._expand_ratio_list)))
        if max(self._expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = Mo.Sequential(OrderedDict([
                ('conv', DynamicConv(max(self._in_channel_list), max_middle_channel, kernel=(1, 1))),
                ('bn', DynamicBatchNorm(max_middle_channel, 4)),
                ('act', build_activation(self._act_func)),
            ]))
        depth_conv_list = [
            ('conv', DynamicDepthwiseConv(max_middle_channel, self._kernel_size_list, self._stride)),
            ('bn', DynamicBatchNorm(max_middle_channel, 4)),
            ('act', build_activation(self._act_func))
        ]
        if self._use_se:
            depth_conv_list.append(('se', DynamicSE(max_middle_channel)))
        self.depth_conv = Mo.Sequential(OrderedDict(depth_conv_list))

        self.point_linear = Mo.Sequential(OrderedDict([
            ('conv', DynamicConv(max_middle_channel, max(self._out_channel_list), kernel=(1, 1))),
            ('bn', DynamicBatchNorm(max(self._out_channel_list), 4)),
        ]))

        self.active_kernel_size = max(self._kernel_size_list)
        self.active_expand_ratio = max(self._expand_ratio_list)
        self.active_out_channel = max(self._out_channel_list)

    def call(self, x):
        in_channel = x.shape[1]

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = \
                make_divisible(round(in_channel * self.active_expand_ratio))

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    def extra_repr(self):
        return get_extra_repr(self)

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        importance = np.sum(np.abs(self.point_linear.conv.conv._W.d), axis=(0, 2, 3))
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
        self.point_linear.conv.conv._W.d = np.stack(
            [self.point_linear.conv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv._W.d = np.stack(
            [self.depth_conv.conv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)

        if self._use_se:
            # se expand
            se_expand = self.depth_conv.se.fc.expand
            se_expand._W.d = np.stack([se_expand._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)
            se_expand._b.d = np.stack([se_expand._b.d[idx] for idx in sorted_idx])
            # se reduce
            se_reduce = self.depth_conv.se.fc.reduce
            se_reduce._W.d = np.stack([se_reduce._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
            # middle weight reorganize
            se_importance = np.sum(np.abs(se_expand._W.d), axis=(0, 2, 3))
            se_idx = np.argsort(-se_importance)

            se_expand._W.d = np.stack([se_expand._W.d[:, idx, :, :] for idx in se_idx], axis=1)
            se_reduce._W.d = np.stack([se_reduce._W.d[idx, :, :, :] for idx in se_idx], axis=0)
            se_reduce._b.d = np.stack([se_reduce._b.d[idx] for idx in se_idx])

        if self.inverted_bottleneck is not None:
            adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv._W.d = np.stack(
                [self.inverted_bottleneck.conv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)
            return None
        else:
            return sorted_idx

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
        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv._W.d = \
                self.inverted_bottleneck.conv.conv._W.d[:middle_channel, :in_channel, :, :]
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        active_filter = self.depth_conv.conv.get_active_filter(middle_channel, self.active_kernel_size)
        sub_layer.depth_conv.conv._W.d = active_filter.d
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self._use_se:
            se_mid = make_divisible(middle_channel // SEModule.REDUCTION)
            sub_layer.depth_conv.se.fc.reduce._W.d =\
                self.depth_conv.se.fc.reduce._W.d[:se_mid, :middle_channel, :, :]
            sub_layer.depth_conv.se.fc.reduce._b.d =\
                self.depth_conv.se.fc.reduce._b.d[:se_mid]

            sub_layer.depth_conv.se.fc.expand._W.d =\
                self.depth_conv.se.fc.expand._W.d[:middle_channel, :se_mid, :, :]
            sub_layer.depth_conv.se.fc.expand._b.d =\
                self.depth_conv.se.fc.expand._b.d[:middle_channel]

        sub_layer.point_linear.conv._W.d =\
            self.point_linear.conv.conv._W.d[:self.active_out_channel, :middle_channel, :, :]
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)
        nn.set_auto_forward(False)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            'name': MBConvLayer.__name__,
            'in_channels': in_channel,
            'out_channels': self.active_out_channel,
            'kernel': (self.active_kernel_size, self.active_kernel_size),
            'stride': self._stride,
            'expand_ratio': self.active_expand_ratio,
            'mid_channels': self.active_middle_channel(in_channel),
            'act_func': self._act_func,
            'use_se': self._use_se,
        }


class DynamicMiddleFlowXPBlock(Mo.Module):

    r"""Dynamic Middle Flow Xception Block

    This block implements the dynamic version of MiddleFlow blocks for
    Xception.

    Args:
        in_channel_list (list of int): Candidates for the number of
            active input channels.
        out_channel_list (list of int): Candidates for the number of
            output channels.
        kernel_size_list (list of int or int): Candidates for the kernel size.
            Defaults to 3.
        expand_ratio_list (list of int or int): Candidates for the expand
            ratio. Defaults to 1.
        stride (tuple of int, optional): Stride sizes for dimensions.
            Defaults to (1, 1).
        depth (int, optional): Initialise the runtime depth.
            Defaults to 3 (which is the maximum depth).
    """

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=1, stride=(1, 1), depth=3):
        self._in_channel_list = in_channel_list
        self._out_channel_list = out_channel_list
        self._kernel_size_list = val2list(kernel_size_list)
        self._expand_ratio_list = val2list(expand_ratio_list)
        self._stride = stride

        # build modules
        max_middle_channel = make_divisible(
            round(max(self._in_channel_list) * max(self._expand_ratio_list)))

        self._depth_conv1 = Mo.Sequential(OrderedDict([
            ('act', build_activation('relu')),
            ('dwconv', DynamicDepthwiseConv(max(self._in_channel_list), self._kernel_size_list, self._stride)),
        ]))

        self._point_linear1 = Mo.Sequential(OrderedDict([
            ('ptconv', DynamicConv(max(self._in_channel_list), max_middle_channel, kernel=(1, 1))),
            ('bn', DynamicBatchNorm(max_middle_channel, 4))
        ]))

        self._depth_conv2 = Mo.Sequential(OrderedDict([
            ('act', build_activation('relu')),
            ('dwconv', DynamicDepthwiseConv(max_middle_channel, self._kernel_size_list, self._stride)),
        ]))

        self._point_linear2 = Mo.Sequential(OrderedDict([
            ('ptconv', DynamicConv(max_middle_channel, max_middle_channel, kernel=(1, 1))),
            ('bn', DynamicBatchNorm(max_middle_channel, 4))
        ]))

        self._depth_conv3 = Mo.Sequential(OrderedDict([
            ('act', build_activation('relu')),
            ('dwconv', DynamicDepthwiseConv(max_middle_channel, self._kernel_size_list, self._stride)),
        ]))

        self._point_linear3 = Mo.Sequential(OrderedDict([
            ('ptconv', DynamicConv(max_middle_channel, max(self._out_channel_list), kernel=(1, 1))),
            ('bn', DynamicBatchNorm(max(self._out_channel_list), 4))
        ]))

        self.runtime_depth = depth
        self.active_kernel_size = max(self._kernel_size_list)
        self.active_expand_ratio = max(self._expand_ratio_list)
        self.active_out_channel = max(self._out_channel_list)

    def call(self, inp):
        in_channel = inp.shape[1]

        self._depth_conv1.dwconv.active_kernel_size = self.active_kernel_size
        self._point_linear1.ptconv.active_out_channel = \
            make_divisible(round(in_channel * self.active_expand_ratio))

        self._depth_conv2.dwconv.active_kernel_size = self.active_kernel_size
        self._point_linear2.ptconv.active_out_channel = \
            make_divisible(round(in_channel * self.active_expand_ratio))

        self._depth_conv3.dwconv.active_kernel_size = self.active_kernel_size
        self._point_linear3.ptconv.active_out_channel = self.active_out_channel

        if self.runtime_depth == 1:
            self._point_linear1.ptconv.active_out_channel = self.active_out_channel
        elif self.runtime_depth == 2:
            self._point_linear2.ptconv.active_out_channel = self.active_out_channel

        x = self._depth_conv1(inp)
        x = self._point_linear1(x)

        if self.runtime_depth > 1:  # runtime depth
            x = self._depth_conv2(x)
            x = self._point_linear2(x)

        if self.runtime_depth > 2:  # runtime depth
            x = self._depth_conv3(x)
            x = self._point_linear3(x)

        # Skip is a simple shortcut ->
        skip = inp
        x += skip
        return x

    def extra_repr(self):
        return get_extra_repr(self)

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        importance = np.sum(np.abs(self._point_linear3.ptconv.conv._W.d), axis=(0, 2, 3))
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
        self._point_linear3.ptconv.conv._W.d = np.stack(
            [self._point_linear3.ptconv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
        adjust_bn_according_to_idx(self._point_linear3.bn.bn, sorted_idx)

        self._point_linear2.ptconv.conv._W.d = np.stack(
            [self._point_linear2.ptconv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
        adjust_bn_according_to_idx(self._point_linear2.bn.bn, sorted_idx)

        self._point_linear1.ptconv.conv._W.d = np.stack(
            [self._point_linear1.ptconv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
        adjust_bn_according_to_idx(self._point_linear1.bn.bn, sorted_idx)

        self._depth_conv3.dwconv.conv._W.d = np.stack(
            [self._depth_conv3.dwconv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)

        self._depth_conv2.dwconv.conv._W.d = np.stack(
            [self._depth_conv2.dwconv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)

        self._depth_conv1.dwconv.conv._W.d = np.stack(
            [self._depth_conv1.dwconv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)

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

        active_filter = self._depth_conv1.dwconv.get_active_filter(in_channel, self.active_kernel_size)
        sub_layer.rep.sepconv1.dwconv._W.d = active_filter.d
        copy_bn(sub_layer.rep.sepconv1.bn, self._point_linear1.bn.bn)

        if self.runtime_depth == 1:
            sub_layer.rep.sepconv1.pointwise._W.d =\
                self._point_linear1.ptconv.conv._W.d[:self.active_out_channel, :in_channel, :, :]

        if self.runtime_depth > 1:
            active_filter = self._depth_conv2.dwconv.get_active_filter(middle_channel, self.active_kernel_size)
            sub_layer.rep.sepconv2.dwconv._W.d = active_filter.d

            copy_bn(sub_layer.rep.sepconv2.bn, self._point_linear2.bn.bn)

            sub_layer.rep.sepconv1.pointwise._W.d =\
                self._point_linear1.ptconv.conv._W.d[:middle_channel, :in_channel, :, :]

        if self.runtime_depth == 2:
            sub_layer.rep.sepconv2.pointwise._W.d =\
                self._point_linear2.ptconv.conv._W.d[:self.active_out_channel, :middle_channel, :, :]

        if self.runtime_depth > 2:
            active_filter = self._depth_conv3.dwconv.get_active_filter(middle_channel, self.active_kernel_size)
            sub_layer.rep.sepconv3.dwconv._W.d = active_filter.d
            copy_bn(sub_layer.rep.sepconv3.bn, self._point_linear3.bn.bn)

            sub_layer.rep.sepconv2.pointwise._W.d =\
                self._point_linear2.ptconv.conv._W.d[:middle_channel, :middle_channel, :, :]

            sub_layer.rep.sepconv3.pointwise._W.d =\
                self._point_linear3.ptconv.conv._W.d[:self.active_out_channel, :middle_channel, :, :]

        nn.set_auto_forward(False)

        return sub_layer

    def get_active_subnet_config(self, in_channels):
        return {
            'name': XceptionBlock.__name__,
            'in_channels': in_channels,
            'out_channels': self.active_out_channel,
            'reps': self.runtime_depth,
            'kernel': (self.active_kernel_size, self.active_kernel_size),
            'stride': self._stride,
            'expand_ratio': self.active_expand_ratio,
        }


class DynamicBottleneckResidualBlock(Mo.Module):

    r"""Dynamic BottleneckResidualBlock
    This block implements the dynamic version of bottleneck blocks of
    ResNet.

    Args:
        in_channel_list (list of int): Candidates for the number of
            active input channels.
        out_channel_list (list of int): Candidates for the number of
            output channels.
        expand_ratio_list (list of float or float, optional): Candidates
            for the expand ratio. Defaults to 0.25.
        kernel (tuple of int, optional): Kernel size. Defaults to (3, 3).
        stride (tuple of int, optional): Stride sizes for dimensions.
            Defaults to (1, 1).
        act_func (str, optional) Type of activation. Defaults to 'relu'.
        downsample_mode (str, optional): Downsample method for the
           residual connection. Defaults to 'avgpool_conv'.
    """

    def __init__(self, in_channel_list, out_channel_list, expand_ratio_list=0.25,
                 kernel=(3, 3), stride=(1, 1), act_func='relu', downsample_mode='avgpool_conv'):

        self._in_channel_list = in_channel_list
        self._out_channel_list = out_channel_list
        self._expand_ratio_list = val2list(expand_ratio_list)

        self._kernel = kernel
        self._stride = stride
        self._act_func = act_func
        self._downsample_mode = downsample_mode

        # build modules
        max_middle_channel = make_divisible(
            round(max(self._out_channel_list) * max(self._expand_ratio_list)), 8)

        self.conv1 = Mo.Sequential(OrderedDict([
            ('conv', DynamicConv(max(self._in_channel_list), max_middle_channel, (1, 1))),
            ('bn', DynamicBatchNorm(max_middle_channel, 4)),
            ('act', build_activation(self._act_func))
        ]))
        self.conv2 = Mo.Sequential(OrderedDict([
            ('conv', DynamicConv(max_middle_channel, max_middle_channel, kernel, stride=stride)),
            ('bn', DynamicBatchNorm(max_middle_channel, 4)),
            ('act', build_activation(self._act_func))
        ]))
        self.conv3 = Mo.Sequential(OrderedDict([
            ('conv', DynamicConv(max_middle_channel, max(self._out_channel_list), (1, 1))),
            ('bn', DynamicBatchNorm(max(self._out_channel_list), 4)),
        ]))

        if self._stride == (1, 1) and self._in_channel_list == self._out_channel_list:
            self.downsample = Mo.Identity()
        elif self._downsample_mode == 'conv':
            self.downsample = Mo.Sequential(OrderedDict([
                ('conv', DynamicConv(max(self._in_channel_list), max(self._out_channel_list), stride=stride)),
                ('bn', DynamicBatchNorm(max(self._out_channel_list), 4)),
            ]))
        elif self._downsample_mode == 'avgpool_conv':
            self.downsample = Mo.Sequential(OrderedDict([
                ('avg_pool', Mo.AvgPool(stride, stride=stride, pad=(0, 0), ignore_border=False)),
                ('conv', DynamicConv(max(self._in_channel_list), max(self._out_channel_list), (1, 1))),
                ('bn', DynamicBatchNorm(max(self._out_channel_list), 4)),
            ]))
        else:
            raise NotImplementedError

        self.final_act = build_activation(self._act_func)

        self.active_expand_ratio = max(self._expand_ratio_list)
        self.active_out_channel = max(self._out_channel_list)

    def call(self, x):
        feature_dim = self.active_middle_channels

        self.conv1.conv.active_out_channel = feature_dim
        self.conv2.conv.active_out_channel = feature_dim
        self.conv3.conv.active_out_channel = self.active_out_channel
        if not isinstance(self.downsample, Mo.Identity):
            self.downsample.conv.active_out_channel = self.active_out_channel

        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + residual
        x = self.final_act(x)

        return x

    def extra_repr(self):
        return get_extra_repr(self)

    @property
    def active_middle_channels(self):
        feature_dim = round(self.active_out_channel * self.active_expand_ratio)
        feature_dim = make_divisible(feature_dim)
        return feature_dim

    def re_organize_middle_weights(self, expand_ratio_stage):
        # conv3 -> conv2
        importance = np.sum(np.abs(self.conv3.conv.conv._W.d), axis=(0, 2, 3))
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self._expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self._out_channel_list) * expand))
                for expand in sorted_expand_list
            ]
            right = len(importance)
            base = - len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                if right > left:  # not in the original code
                    importance[left:right] += base
                    base += 1e5
                    right = left
        sorted_idx = np.argsort(-importance)
        self.conv3.conv.conv._W.d = np.stack([self.conv3.conv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
        adjust_bn_according_to_idx(self.conv2.bn.bn, sorted_idx)
        self.conv2.conv.conv._W.d = np.stack([self.conv2.conv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)

        # conv2 -> conv1
        importance = np.sum(np.abs(self.conv2.conv.conv._W.d), axis=(0, 2, 3))
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self._expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self._out_channel_list) * expand))
                for expand in sorted_expand_list
            ]
            right = len(importance)
            base = - len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                if right > left:  # not in original code, need to check
                    importance[left:right] += base
                    base += 1e5
                    right = left
        sorted_idx = np.argsort(-importance)
        self.conv2.conv.conv._W.d = np.stack([self.conv2.conv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
        adjust_bn_according_to_idx(self.conv1.bn.bn, sorted_idx)
        self.conv1.conv.conv._W.d = np.stack([self.conv1.conv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)

        return None

    def get_active_subnet(self, in_channel, preserve_weight=True):
        with nn.auto_forward(True):
            # build the new layer
            sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
            if not preserve_weight:
                return sub_layer

            # copy weight from current layer
            active_filter = self.conv1.conv.get_active_filter(self.active_middle_channels, in_channel)
            sub_layer.conv1.conv._W.d = active_filter.d
            copy_bn(sub_layer.conv1.bn, self.conv1.bn.bn)

            active_filter = self.conv2.conv.get_active_filter(self.active_middle_channels, self.active_middle_channels)
            sub_layer.conv2.conv._W.d = active_filter.d
            copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

            active_filter = self.conv3.conv.get_active_filter(self.active_out_channel, self.active_middle_channels)
            sub_layer.conv3.conv._W.d = active_filter.d
            copy_bn(sub_layer.conv3.bn, self.conv3.bn.bn)

            if not isinstance(self.downsample, Mo.Identity):
                active_filter = self.downsample.conv.get_active_filter(self.active_out_channel, in_channel)
                sub_layer.downsample.conv._W.d = active_filter.d
                copy_bn(sub_layer.downsample.bn, self.downsample.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            'name': BottleneckResidualBlock.__name__,
            'in_channels': in_channel,
            'out_channels': self.active_out_channel,
            'kernel': self._kernel,
            'stride': self._stride,
            'expand_ratio': self.active_expand_ratio,
            'mid_channels': self.active_middle_channels,
            'act_func': self._act_func,
            'downsample_mode': self._downsample_mode,
        }
