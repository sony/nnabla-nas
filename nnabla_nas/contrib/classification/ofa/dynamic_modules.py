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
import nnabla.functions as F

from .... import module as Mo
#from .utils import *
from .modules import MBConvLayer, ConvLayer, IdentityLayer, set_layer_from_config
from .ofa_modules.common_tools import val2list, make_divisible
from .ofa_modules.pytorch_modules import SEModule
from .ofa_modules.dynamic_op import DynamicConv2d, DynamicBatchNorm2d, DynamicSeparableConv2d, DynamicLinear, DynamicSE
from .ofa_modules.my_modules import MyNetwork
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

	def __init__(self, in_channel_list, out_channel_list, kernel=(3,3), stride=(1,1), dilation=(1,1),
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
	
	def call(self, input):
		self.conv.active_out_channel = self.active_out_channel
		x = self.conv(input)
		if self._use_bn:
			x = self.bn(x)
		x = self.act(x)
		return x
	
	@property
	def module_str(self):
		return 'DyConv(O%d, K%d, S%d)' % (self.active_out_channel, self._kernel[0], self._stride[0])
	
	@property
	def get_active_parameter_size(self):
		param_W = np.prod(self.conv._W[:self.active_out_channel, :, :, :].shape)
		param_b = np.prod(self.conv._b[:self.active_out_channel].shape)
		if self._use_bn:
			param_bn = np.prod(self.bn._beta[:, :self.active_out_channel, :, :].shape)
			return param_W + param_b + param_bn * 4
		else:
			return param_W + param_b

class DynamicLinearLayer(Mo.Module):

	def __init__(self, in_features_list, out_features, bias=True, dropout_rate=0):

		self._in_features_list = in_features_list
		self._out_features = out_features
		self._bias = bias
		self._dropout_rate = dropout_rate

		if self._dropout_rate > 0:
			self.dropout = Mo.Dropout(self._dropout_rate)
		else:
			self.dropout = None
		self.linear = DynamicLinear(
			max(self._in_features_list), self._out_features, bias=self._bias)

	def call(self, x):
		if self.dropout is not None:
			x = self.dropout(x)
		return self.linear(x)
	
	@property
	def module_str(self):
		return 'DyLinear(%d, %d)' % (max(self._in_features_list), self._out_features)
	
	@property
	def get_active_parameter_size(self):
		param_W = np.prod(self.linear._W[:self._out_features, :].shape)
		param_b = np.prod(self.linear._b[:self._out_features].shape)
		return param_W + param_b
		

class DynamicMBConvLayer(Mo.Module):

	def __init__(self, in_channel_list, out_channel_list, 
				kernel_size_list=3, expand_ratio_list=6, stride=(1,1), 
				act_func='relu6', use_se=False):

		self._in_channel_list = in_channel_list
		self._out_channel_list = out_channel_list
		self._kernel_size_list = val2list(kernel_size_list)
		self._expand_ratio_list = val2list(expand_ratio_list)
		self._stride = stride
		self._act_func = act_func
		self._use_se = use_se

		# [build modules]
		max_middle_channel = make_divisible(
			round(max(self._in_channel_list) * max(self._expand_ratio_list)))
		if max(self._expand_ratio_list) == 1:
			self.inverted_bottleneck = None
		else:
			self.inverted_bottleneck = Mo.Sequential(OrderedDict([
				('conv', DynamicConv2d(max(self._in_channel_list), max_middle_channel)),
				('bn', DynamicBatchNorm2d(max_middle_channel, 4)),
				('act', build_activation(self._act_func)),
			]))
		depth_conv_list = [
			('conv', DynamicSeparableConv2d(max_middle_channel, self._kernel_size_list, self._stride)),
			('bn', DynamicBatchNorm2d(max_middle_channel, 4)),
			('act', build_activation(self._act_func))
		]
		if self._use_se:
			depth_conv_list.append(('se', DynamicSE(max_middle_channel)))
		self.depth_conv = Mo.Sequential(OrderedDict(depth_conv_list))

		self.point_linear = Mo.Sequential(OrderedDict([
			('conv', DynamicConv2d(max_middle_channel, max(self._out_channel_list))),
			('bn', DynamicBatchNorm2d(max(self._out_channel_list), 4)),
		]))

		self.active_kernel_size = max(self._kernel_size_list)
		self.active_expand_ratio = max(self._expand_ratio_list)
		self.active_out_channel = max(self._out_channel_list)

	def call(self, input):
		in_channel = input.shape[1]
		
		if self.inverted_bottleneck is not None:
			self.inverted_bottleneck.conv.active_out_channel = \
				make_divisible(round(in_channel * self.active_expand_ratio))

		self.depth_conv.conv.active_kernel_size = self.active_kernel_size
		self.point_linear.conv.active_out_channel = self.active_out_channel

		if self.inverted_bottleneck is not None:
			input = self.inverted_bottleneck(input)
		input = self.depth_conv(input)
		input = self.point_linear(input)
		return input

	def re_organize_middle_weights(self, expand_ratio_stage=0):
		importance = np.sum(np.abs(self.point_linear.conv.conv._W.d), axis=(0, 2, 3))
		if expand_ratio_stage > 0: # totally not sure about this operation
			sorted_expand_list = copy.deepcopy(self._expand_ratio_list)
			sorted_expand_list.sort(reverse=True)
			target_width_list = [
				make_divisible(round(max(self._in_channel_list) * expand))
				for expand in sorted_expand_list
			]
			right = len(importance)
			base = - len(target_width_list) * 1e5
			for i in range(expand_ratio_stage + 1): #not sure
				left = target_width_list[i]
				#print(left, right)
				if right > left: # not in the original code
					importance[left:right] += base
					base += 1e5
					right = left

		sorted_idx = np.argsort(-importance)
		self.point_linear.conv.conv._W.d = np.stack(
			[self.point_linear.conv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
		adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
		self.depth_conv.conv.conv._W.d = np.stack(
			[self.depth_conv.conv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)
		
		if self._use_se:
			# [se expand]
			se_expand = self.depth_conv.se.fc.expand
			se_expand._W.d = np.stack([se_expand._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)
			se_expand._b.d = np.stack([se_expand._b.d[idx] for idx in sorted_idx])
			# [se reduce]
			se_reduce = self.depth_conv.se.fc.reduce
			se_reduce._W.d = np.stack([se_reduce._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
			# [middle weight reorganize]
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
				#self.inverted_bottleneck.conv.get_active_filter(middle_channel, in_channel).data
			copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

		active_filter = self.depth_conv.conv.get_active_filter(middle_channel, self.active_kernel_size)
		sub_layer.depth_conv.conv._W.d = active_filter.d
			#self.depth_conv.conv.get_active_filter(middle_channel, self.active_kernel_size).data
		copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

		if self._use_se:
			se_mid = make_divisible(middle_channel // SEModule.REDUCTION)
			sub_layer.depth_conv.se.fc.reduce._W.d =\
				self.depth_conv.se.fc.reduce._W.d[:se_mid, :middle_channel, :, :]
				#self.depth_conv.se.get_active_reduce_weight(se_mid, middle_channel).d
			sub_layer.depth_conv.se.fc.reduce._b.d =\
				self.depth_conv.se.fc.reduce._b.d[:se_mid]
				#self.depth_conv.se.get_active_reduce_bias(se_mid).d

			sub_layer.depth_conv.se.fc.expand._W.d =\
				self.depth_conv.se.fc.expand._W.d[:middle_channel, :se_mid, :, :]
				#self.depth_conv.se.get_active_expand_weight(se_mid, middle_channel).d
			sub_layer.depth_conv.se.fc.expand._b.d =\
				self.depth_conv.se.fc.expand._b.d[:middle_channel]
				#self.depth_conv.se.get_active_expand_bias(middle_channel).d

		sub_layer.point_linear.conv._W.d =\
			self.point_linear.conv.conv._W.d[:self.active_out_channel, :middle_channel, :, :]
			#self.point_linear.conv.get_active_filter(self.active_out_channel, middle_channel).data
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
	

class DynamicResNetBottleneckBlock(Mo.Module):
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
			('conv', DynamicConv2d(max(self._in_channel_list), max_middle_channel)),
			('bn', DynamicBatchNorm2d(max_middle_channel, 4)),
			('act', build_activation(self._act_func)),
		]))
		self.conv2 = Mo.Sequential(OrderedDict([
			('conv', DynamicConv2d(max_middle_channel, max_middle_channel, kernel, stride=stride)),
			('bn', DynamicBatchNorm2d(max_middle_channel, 4)),
			('act', build_activation(self._act_func)),
		]))
		self.conv3 = Mo.Sequential(OrderedDict([
			('conv', DynamicConv2d(max_middle_channel, max(self._out_channel_list))),
			('bn', DynamicBatchNorm2d(max(self._out_channel_list), 4)),
		]))

		if self._stride == (1, 1) and self._in_channel_list == self._out_channel_list:
			self.downsample = IdentityLayer(max(self._in_channel_list), max(self._out_channel_list))
		elif self._downsample_mode == 'conv':
			self.downsample = Mo.Sequential(OrderedDict([
				('conv', DynamicConv2d(max(self._in_channel_list), max(self._out_channel_list), stride=stride)),
				('bn', DynamicBatchNorm2d(max(self._out_channel_list), 4)),
			]))
		elif self._downsample_mode == 'avgpool_conv':
			self.downsample = Mo.Sequential(OrderedDict([
				('avg_pool', Mo.AvgPool(kernel=stride, stride=stride, pad=(0, 0))),
				('conv', DynamicConv2d(max(self._in_channel_list), max(self._out_channel_list))),
				('bn', DynamicBatchNorm2d(max(self._out_channel_list), 4)),
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
		if not isinstance(self.downsample, IdentityLayer):
			self.downsample.conv.active_out_channel = self.active_out_channel

		residual = self.downsample(x)

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)

		x = x + residual
		x = self.final_act(x)

		return x
	
	@property
	def module_str(self):
		return '(%s, %s)' % (
			'%dx%d_BottleneckConv_in->%d->%d_S%d' % (
				self._kernel[0], self._kernel[0], self.active_middle_channels, 
				self.active_out_channel, self._stride[0]
			),
			'Identity' if isinstance(self.downsample, Mo.Identity) else self._downsample_mode,
		)
	
	@property
	def get_active_parameter_size(self):
		def get_active_params_for_conv(_conv, active_channels):
			param_W = np.prod(_conv.conv._W[:active_channels, :, :, :].shape)
			param_b = np.prod(_conv.conv._b[:active_channels].shape)
			param_bn = np.prod(_conv.bn._beta[:, :active_channels, :, :].shape)
			return param_W + param_b + param_bn * 4
		params = get_active_params_for_conv(self.conv1, self.active_middle_channels)
		params += get_active_params_for_conv(self.conv2, self.active_middle_channels)
		params += get_active_params_for_conv(self.conv3, self.active_out_channel)
		return params
	
	@property
	def active_middle_channels(self):
		feature_dim = round(self.active_out_channel * self.active_expand_ratio)
		feature_dim = make_divisible(feature_dim)
		return feature_dim

	def re_organize_middle_weights(self, expand_ratio_stage):
		# conv3 -> conv2
		importance = np.sum(np.abs(self.conv3.conv.conv._W.d), axis=(0, 2, 3))
		if expand_ratio_stage > 0: # totally not sure about this operation
			sorted_expand_list = copy.deepcopy(self._expand_ratio_list)
			sorted_expand_list.sort(reverse=True)
			target_width_list = [
				make_divisible(round(max(self._out_channel_list) * expand))
				for expand in sorted_expand_list
			]
			right = len(importance)
			base = - len(target_width_list) * 1e5
			for i in range(expand_ratio_stage + 1): #not sure
				left = target_width_list[i]
				#print(left, right)
				if right > left: # not in the original code
					importance[left:right] += base
					base += 1e5
					right = left
		#_, sorted_idx = F.sort(importance, reverse=True, with_index=True)
		sorted_idx = np.argsort(-importance)
		self.conv3.conv.conv._W.d = np.stack([self.conv3.conv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
		adjust_bn_according_to_idx(self.conv2.bn.bn, sorted_idx)
		self.conv2.conv.conv._W.d = np.stack([self.conv2.conv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)
		
		# conv2 -> conv1
		importance = np.sum(np.abs(self.conv2.conv.conv._W.d), axis=(0, 2, 3))
		if expand_ratio_stage > 0: # totally not sure about this operation
			sorted_expand_list = copy.deepcopy(self._expand_ratio_list)
			sorted_expand_list.sort(reverse=True)
			target_width_list = [
				make_divisible(round(max(self._out_channel_list) * expand), MyNetwork.CHANNEL_DIVISIBLE)
				for expand in sorted_expand_list
			]
			right = len(importance)
			base = - len(target_width_list) * 1e5
			for i in range(expand_ratio_stage + 1): #not sure
				left = target_width_list[i]
				#print(left, right)
				if right > left: # not in original code, need to check
					importance[left:right] += base
					base += 1e5
					right = left
		sorted_idx = np.argsort(-importance)
		self.conv2.conv.conv._W.d = np.stack([self.conv2.conv.conv._W.d[:, idx, :, :] for idx in sorted_idx], axis=1)
		adjust_bn_according_to_idx(self.conv1.bn.bn, sorted_idx)
		self.conv1.conv.conv._W.d = np.stack([self.conv1.conv.conv._W.d[idx, :, :, :] for idx in sorted_idx], axis=0)

		return None



