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
from collections import Counter, OrderedDict
import copy

import nnabla as nn
import nnabla.functions as F
import numpy as np
import os
import random

from ..base import ClassificationModel as Model
from .modules import ResidualBlock, ConvLayer, IdentityLayer, LinearLayer, MBConvLayer
from .dynamic_modules import DynamicMBConvLayer
from .ofa_modules.my_modules import MyGlobalAvgPool2d
from .ofa_modules.common_tools import val2list, make_divisible
from .ofa_modules.common_tools import label_smoothing_loss, cross_entropy_loss_with_label_smoothing
from .... import module as Mo
from .networks.mobilenet_v3 import MobileNetV3

class SearchNet(MobileNetV3):
	r"""
	Reference:
	https://github.com/mit-han-lab/once-for-all/blob/master/ofa/imagenet_classification/elastic_nn/networks/ofa_mbv3.py
	"""

	def __init__(self,
				 num_classes=1000,
				 bn_param=(0.9, 1e-5),
				 dropout=0.1,
				 base_stage_width=None,
				 width_mult=1.0,
				 ks_list=3,
				 expand_ratio_list=6,
				 depth_list=4
				 ):

		self._num_classes = num_classes
		self._bn_param = bn_param
		self._dropout = dropout
		#self._arch_idx = None # keeps current max arch

		self._width_mult = width_mult
		self._ks_list = val2list(ks_list, 1)
		self._expand_ratio_list = val2list(expand_ratio_list, 1)
		self._depth_list = val2list(depth_list)

		# sort
		self._ks_list.sort()
		self._expand_ratio_list.sort()
		self._depth_list.sort()

		base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

		final_expand_width = make_divisible(base_stage_width[-2] * self._width_mult)
		last_channel = make_divisible(base_stage_width[-1] * self._width_mult)

		stride_stages = [1, 2, 2, 2, 1, 2]
		act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
		se_stages = [False, False, True, False, True, True]
		n_block_list = [1] + [max(self._depth_list)] * 5
		width_list = []
		for base_width in base_stage_width[:-2]:
			width = make_divisible(base_width * self._width_mult)
			width_list.append(width)
		
		input_channel, first_block_dim = width_list[0], width_list[1]
		# [first conv layer]
		first_conv = ConvLayer(
			3, input_channel, kernel=(3, 3), stride=(2, 2), act_func='h_swish')
		first_block_conv = MBConvLayer(
			input_channel, first_block_dim, kernel=(3, 3), stride=(stride_stages[0], stride_stages[0]),
			expand_ratio=1, act_func=act_stages[0], use_se=se_stages[0],
		)
		first_block = ResidualBlock(
			first_block_conv,
			IdentityLayer(first_block_dim, first_block_dim) if input_channel == first_block_dim else None,
		)

		# [inverted residual blocks]
		self.block_group_info = []
		blocks = [first_block]
		_block_index = 1
		feature_dim = first_block_dim
		for width, n_block, s, act_func, use_se in zip(width_list[2:], n_block_list[1:],
													   stride_stages[1:], act_stages[1:], se_stages[1:]):
			self.block_group_info.append([_block_index + i for i in range(n_block)])
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
					kernel_size_list=ks_list, expand_ratio_list=expand_ratio_list,
					stride=stride, act_func=act_func, use_se=use_se,
				)
				if stride == (1, 1) and feature_dim == output_channel:
					shortcut = IdentityLayer(feature_dim, feature_dim)
				else:
					shortcut = None
				blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
				feature_dim = output_channel

		# [final expand layer, feature mix layer & classifier]
		final_expand_layer = ConvLayer(
			feature_dim, final_expand_width, kernel=(1, 1), act_func='h_swish'
		)
		feature_mix_layer = ConvLayer(
			final_expand_width, last_channel, kernel=(1, 1), 
			with_bias=False, use_bn=False, act_func='h_swish'
		)
		classifier = LinearLayer(last_channel, num_classes, dropout=dropout)
		
		super(SearchNet, self).__init__(
			first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)

		# [set bn param]
		self.set_bn_param(decay_rate=bn_param[0], eps=bn_param[1])

		# [runtime depth]
		self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

	def call(self, x):
		x = self.first_conv(x)
		x = self.blocks[0](x)
		# [blocks]
		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			for idx in active_idx:
				x = self.blocks[idx](x)
		x = self.final_expand_layer(x)
		#x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
		#x = self.global_avg_pool(x)
		x = F.mean(x, axis=(2, 3), keepdims=True)
		x = self.feature_mix_layer(x)
		#x = x.view(x.size(0), -1)
		x = F.reshape(x, shape=(x.shape[0], -1))
		return self.classifier(x)
	
	@property
	def module_str(self):
		_str = self.first_conv.module_str + '\n'
		_str += self.blocks[0].module_str + '\n'

		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			for idx in active_idx:
				_str += self.blocks[idx].module_str + '\n'

		_str += self.final_expand_layer.module_str + '\n'
		_str += self.feature_mix_layer.module_str + '\n'
		_str += self.classifier.module_str + '\n'
		return _str

	@property
	def grouped_block_index(self):
		return self.block_group_info

	def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
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
		ks_candidates = self._ks_list if self.__dict__.get('_ks_include_list', None) is None \
			else self.__dict__['_ks_include_list']
		expand_candidates = self._expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
			else self.__dict__['_expand_include_list']
		depth_candidates = self._depth_list if self.__dict__.get('_depth_include_list', None) is None else \
			self.__dict__['_depth_include_list']
		
		# [sample kernel size]
		ks_setting = []
		if not isinstance(ks_candidates[0], list):
			ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
		for k_set in ks_candidates:
			k = random.choice(k_set)
			ks_setting.append(k)
		
		# [sample expand ratio]
		expand_setting = []
		if not isinstance(expand_candidates[0], list):
			expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
		for e_set in expand_candidates:
			e = random.choice(e_set)
			expand_setting.append(e)

		# [sample depth]
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

	def get_active_subnet(self, preserve_weight=True):
		first_conv = self.first_conv
		blocks = [self.blocks[0]]

		final_expand_layer = self.final_expand_layer
		feature_mix_layer = self.feature_mix_layer
		classifier = self.classifier

		input_channel = blocks[0].conv._out_channels
		# blocks
		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			stage_blocks = []
			for idx in active_idx:
				stage_blocks.append(ResidualBlock(
					self.blocks[idx].conv.get_active_subnet(input_channel, preserve_weight),
					self.blocks[idx].shortcut
					#copy.deepcopy(self.blocks[idx].shortcut)
				))
				input_channel = stage_blocks[-1].conv._out_channels
			blocks += stage_blocks

		#super(SearchNet, self).__init__(
		#	first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
		#self.set_bn_param(**self.get_bn_param())
		_subnet = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
		_subnet.set_bn_param(**self.get_bn_param())

		return _subnet

	
	def re_organize_middle_weights(self, expand_ratio_stage=0):
		print("re_organize_middle_weights")
		for block in self.blocks[1:]:
			block.conv.re_organize_middle_weights(expand_ratio_stage)

	def load_ofa_parameters(self, path, raise_if_missing=False):
		with nn.parameter_scope('', OrderedDict()):
			nn.load_parameters(path)
			params = nn.get_parameters(grad_only=False)
		self.set_ofa_parameters(params, raise_if_missing=raise_if_missing)
		
	def set_ofa_parameters(self, params, raise_if_missing=False):
		for prefix, module in self.get_modules():
			for name, p in module.parameters.items():
				key = prefix + ('/' if prefix else '') + name
				if '/mobile_inverted_conv/' in key:
					new_key = key.replace('/mobile_inverted_conv/', '/conv/')
				else:
					new_key = key
				if new_key in params:
					pass
				elif '/bn/bn/' in new_key:
					new_key = new_key.replace('/bn/bn/', '/bn/')
				elif '/conv/conv/_W' in new_key:
					new_key = new_key.replace('/conv/conv/_W', '/conv/_W')
				elif '/linear/linear/' in new_key:
					new_key = new_key.replace('/linear/linear/', '/linear/')
				##############################################################################
				elif '/linear/' in new_key:
					new_key = new_key.replace('/linear/', '/linear/linear/')
				elif 'bn/' in new_key:
					new_key = new_key.replace('bn/', 'bn/bn/')
				elif 'conv/_W' in new_key:
					new_key = new_key.replace('conv/_W', 'conv/conv/_W')
				else:
					if raise_if_missing:
						raise ValueError(
							f'A child module {name} cannot be found in '
							'{this}. This error is raised because '
							'`raise_if_missing` is specified '
							'as True. Please turn off if you allow it.')
				p.d = params[new_key].d.copy()
				nn.logger.info(f'`{new_key}` loaded.')

	def extra_repr(self):
		 return (
			 	f'num_classes={self._num_classes}, '
				f'width_mult={self._width_mult}, '
				f'dropout={self._dropout}'
				)


	def save_parameters(self, path=None, params=None, grad_only=False):
		# save the architectures
		# if isinstance(self._modified_resnet_features[3]._mixed, Mo.MixedOp):
		#     output_path = os.path.dirname(path)
		#     plot_resnet(self, os.path.join(output_path, 'arch'))

		super().save_parameters(path, params=params, grad_only=grad_only)

		base_path = path[0:path.rindex("/")]
		weight_str = path[path.rindex("/")+1:].split(".")[0]
		cur_epoch = weight_str.split("_")

		# This needs to be fixed. The self._num_blocks is variable now
		# plot_resnet(self, os.path.join(base_path, f'arch_{cur_epoch}'))

	#def loss(self, outputs, targets, loss_weights=None):
	#	return cross_entropy_loss_with_label_smoothing(outputs[0], targets[0])

class TrainNet(SearchNet):
	def __init__(self, num_classes=1000, bn_param=(0.9, 1e-5), dropout=0.1, base_stage_width=None,
				width_mult=1, ks_list=3, expand_ratio_list=6, depth_list=4, ):
		super(TrainNet, self).__init__(
			num_classes, bn_param, dropout, width_mult=width_mult, 
			ks_list=ks_list, expand_ratio_list=expand_ratio_list, depth_list=depth_list)


			  
	
			   