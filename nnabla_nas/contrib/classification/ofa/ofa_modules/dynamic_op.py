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


import numpy as np

import nnabla as nn
import nnabla.functions as F

from ..... import module as Mo
from .common_tools import get_same_padding, sub_filter_start_end, make_divisible

from .pytorch_modules import SEModule
from .my_modules import MyConv2d


class DynamicSE(SEModule):
    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)
        self._scope_name = f'<dynamicse at {hex(id(self))}>'

    def get_active_reduce_weight(self, num_mid, in_channel, group=None):
        if group is None or group == 1:
            return self.fc.reduce._W[:num_mid, :in_channel, :, :]
        else:
            raise NotImplementedError

    def get_active_reduce_bias(self, num_mid):
        return self.fc.reduce._b[:num_mid] if self.fc.reduce._b is not None else None

    def get_active_expand_weight(self, num_mid, in_channel, group=None):
        if group is None or group == 1:
            return self.fc.expand._W[:in_channel, :num_mid, :, :]
        else:
            raise NotImplementedError

    def get_active_expand_bias(self, in_channel, group=None):
        if group is None or group == 1:
            return self.fc.expand._b[:in_channel] if self.fc.expand._b is not None else None
        else:
            raise NotImplementedError

    def call(self, input, group=None):
        in_channel = input.shape[1]
        num_mid = make_divisible(in_channel // self.reduction)
        y = F.mean(input, axis=(2, 3), keepdims=True)
        # reduce
        reduce_filter = self.get_active_reduce_weight(num_mid, in_channel, group=group)
        reduce_bias = self.get_active_reduce_bias(num_mid)
        y = F.convolution(y, reduce_filter, reduce_bias, pad=(0, 0), stride=(1, 1), dilation=(1, 1), group=1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_filter = self.get_active_expand_weight(num_mid, in_channel, group=group)
        expand_bias = self.get_active_expand_bias(in_channel, group=group)
        y = F.convolution(y, expand_filter, expand_bias, pad=(0, 0), stride=(1, 1), dilation=(1, 1), group=1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)
        return input * y


class DynamicConv2d(Mo.Module):
    def __init__(self, max_in_channels, max_out_channels,
                 kernel=(1, 1), stride=(1, 1), dilation=(1, 1)):
        super(DynamicConv2d, self).__init__()

        self._scope_name = f'<dynamicconv2d at {hex(id(self))}>'
        self._max_in_channels = max_in_channels
        self._max_out_channels = max_out_channels
        self._kernel = kernel
        self._stride = stride
        self._dilation = dilation

        self.conv = Mo.Conv(
            self._max_in_channels, self._max_out_channels, self._kernel,
            stride=self._stride, dilation=(1, 1), with_bias=False
        )

        self.active_out_channel = self._max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        return self.conv._W[:out_channel, :in_channel, :, :]

    def call(self, input, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = input.shape[1]
        filters = self.get_active_filter(out_channel, in_channel)
        padding = get_same_padding(self._kernel)
        filters = F.weight_standardization(filters) if isinstance(self.conv, MyConv2d) else filters
        return F.convolution(input, filters, None, pad=padding,
                             stride=self._stride, dilation=self._dilation, group=1)


class DynamicBatchNorm2d(Mo.Module):
    SET_RUNNING_STATISTICS = False
    GET_STATIC_BN = True

    def __init__(self, max_feature_dim, n_dims):
        super(DynamicBatchNorm2d, self).__init__()

        self._scope_name = f'<dynamicbatchnorm2d at {hex(id(self))}>'

        self._max_feature_dim = max_feature_dim
        self.bn = Mo.BatchNormalization(self._max_feature_dim, n_dims)

    @staticmethod
    def bn_forward(x, bn: Mo.BatchNormalization, feature_dim, training):
        if DynamicBatchNorm2d.GET_STATIC_BN or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            sbeta, sgamma = bn._beta[:, :feature_dim, :, :], bn._gamma[:, :feature_dim, :, :]
            smean = nn.Variable(sbeta.shape)
            svar = nn.Variable(sbeta.shape)
            smean.data = bn._mean.data[:, :feature_dim, :, :]
            svar.data = bn._var.data[:, :feature_dim, :, :]
            y = F.batch_normalization(
                x, sbeta, sgamma, smean, svar, batch_stat=training,)
            if training:
                bn._mean = F.concatenate(smean, bn._mean[:, feature_dim:, :, :], axis=1)
                bn._var = F.concatenate(svar, bn._var[:, feature_dim:, :, :], axis=1)
            return y

    def call(self, input):
        feature_dim = input.shape[1]
        y = self.bn_forward(input, self.bn, feature_dim, self.training)
        return y


class DynamicSeparableConv2d(Mo.Module):
    KERNEL_TRANSFORM_MODE = 1

    def __init__(self, max_in_channels, kernel_size_list, stride=(1, 1), dilation=(1, 1)):
        super(DynamicSeparableConv2d, self).__init__()

        self._scope_name = f'<dynamicseperableconv2d at {hex(id(self))}>'
        self._max_in_channels = max_in_channels
        self._kernel_size_list = kernel_size_list
        self._stride = stride
        self._dilation = dilation

        self.conv = Mo.Conv(
            self._max_in_channels, self._max_in_channels,
            kernel=(max(self._kernel_size_list), max(self._kernel_size_list)),
            stride=self._stride, group=self._max_in_channels, with_bias=False)

        self._ks_set = list(set(self._kernel_size_list))
        self._ks_set.sort()
        if self.KERNEL_TRANSFORM_MODE is not None:
            # resister scaling params
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                param_array = np.eye(ks_small ** 2)
                scale_param = Mo.Parameter(param_array.shape, initializer=param_array, need_grad=True)
                scale_params['%s_matrix' % param_name] = scale_param
            for name, param in scale_params.items():
                self.parameters[name] = param
        self.active_kernel_size = max(self._kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self._kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv._W[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv._W[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = F.reshape(_input_filter, (_input_filter.shape[0], _input_filter.shape[1], -1))
                _input_filter = F.reshape(_input_filter, (-1, _input_filter.shape[2]))
                _input_filter = F.affine(_input_filter, self.parameters['%dto%d_matrix' % (src_ks, target_ks)])
                _input_filter = F.reshape(_input_filter, (filters.shape[0], filters.shape[1], target_ks ** 2))
                _input_filter = F.reshape(_input_filter, (filters.shape[0], filters.shape[1], target_ks, target_ks))
                start_filter = _input_filter
            filters = start_filter

        return filters

    def call(self, input, kernel=None):
        if kernel is None:
            kernel = self.active_kernel_size
        in_channel = input.shape[1]

        filters = self.get_active_filter(in_channel, kernel)
        padding = get_same_padding((kernel, kernel))
        filters = F.weight_standardization(filters) if isinstance(self.conv, MyConv2d) else filters
        return F.convolution(
            input, filters, None, pad=padding, stride=self._stride,
            dilation=self._dilation, group=in_channel)


class DynamicLinear(Mo.Module):
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self._scope_name = f'<dynamiclinear at {hex(id(self))}>'
        self._max_in_features = max_in_features
        self._max_out_features = max_out_features
        self._bias = bias

        self.linear = Mo.Linear(self._max_in_features, self._max_out_features, self._bias)

        self.active_out_features = self._max_out_features

    def get_active_weight(self, out_features, in_features):
        return self.linear._W[:out_features, :in_features]

    def get_active_bias(self, out_features):
        return self.linear._b[:out_features] if self._bias else None

    def call(self, input, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = input.shape[1]
        weight = self.get_active_weight(in_features, out_features)
        bias = self.get_active_bias(out_features)
        return F.affine(input, weight, bias)
