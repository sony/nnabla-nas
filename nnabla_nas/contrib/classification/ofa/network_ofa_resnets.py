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

import nnabla as nn
import nnabla.functions as F
import numpy as np
import os
import random

from ..base import ClassificationModel as Model
from .modules import ResidualBlock, IdentityLayer
from .dynamic_modules import DynamicConvLayer, DynamicResNetBottleneckBlock, DynamicLinearLayer
from .ofa_modules.my_modules import MyGlobalAvgPool2d
from .ofa_modules.common_tools import val2list, make_divisible
from .ofa_modules.common_tools import label_smoothing_loss, cross_entropy_loss_with_label_smoothing
from .... import module as Mo
from .networks.resnets import ResNets

class SearchNetOFAResNets(ResNets):
	r"""Resnet bottleneck with dilated conv search space. This search space takes the count of 
	blocks from Resnet-50, 101. The max search space corresponds to Resnet-50 but with dilated bottleneck blocks.
	Stage 1 to Stage 4 is same as original Resnet. Stage 5 and Stage 6 have dilated bottleneck.
	There are skip connects between them.

	This implementation is based on the PyTorch implementation.

	Args:
		num_classes (int, optional): Number of classes. Defaults to 1000.
		candidates (list of str, optional): A list of candicates. Defaults to
											None.

	References:
	[1] https://github.com/mit-han-lab/once-for-all/blob/master/ofa/imagenet_classification/networks/resnets.py
	"""

	def __init__(self,
				 num_classes=1000,
				 bn_param=(0.9, 1e-5),
				 dropout=0,
				 depth_list=2,
				 expand_ratio_list=0.25,
				 width_mult_list=1.0,
				 ):

		self._num_classes = num_classes
		self._bn_param = bn_param
		self._dropout = dropout
		self._arch_idx = None # keeps current max arch

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

		stage_width_list = ResNets.STAGE_WIDTH_LIST.copy()
		for i, width in enumerate(stage_width_list):
			stage_width_list[i] = [
				make_divisible(width * width_mult) for width_mult in self._width_mult_list
			]

		n_block_list = [base_depth + max(self._depth_list) for base_depth in self.BASE_DEPTH_LIST]
		stride_list = [1, 2, 2, 2]

		#self._settings = [
		#	[w, d, s] for (w, d, s) in zip(stage_width_list, depth_list, stride_list) 
		#]
		input_stem = [
			DynamicConvLayer(val2list(3), mid_input_channel, (3, 3), stride=(2, 2), use_bn=True, act_func='relu'),
			ResidualBlock(
				DynamicConvLayer(mid_input_channel, mid_input_channel, (3, 3), stride=(1, 1), use_bn=True, act_func='relu'),
				IdentityLayer(mid_input_channel, mid_input_channel)
			),
			DynamicConvLayer(mid_input_channel, input_channel, (3, 3), stride=(1, 1), use_bn=True, act_func='relu'),
		] 
		#self.input_stem = Mo.ModuleList(input_stem)
		#self.max_pooling = Mo.MaxPool(kernel=(3, 3), stride=(2, 2), pad=(1, 1))

		# blocks
		blocks = []
		for d, width, s in zip(n_block_list, stage_width_list, stride_list):
			for i in range(d):
				stride = (s, s) if i == 0 else (1, 1)
				blocks.append(
					DynamicResNetBottleneckBlock(
						input_channel,
						width, 
						expand_ratio_list=self._expand_ratio_list,
						kernel=(3, 3),
						stride=stride,
						act_func='relu',
						downsample_mode='avgpool_conv')
				)
				input_channel = width
		#self.blocks = Mo.ModuleList(blocks)
		#self.global_avg_pool = MyGlobalAvgPool2d(keep_dims=False)

		# classifier
		classifier = DynamicLinearLayer(input_channel, num_classes, dropout_rate=dropout)

		super(SearchNetOFAResNets, self).__init__(input_stem, blocks, classifier)
		
		# set bn param
		self.set_bn_param(*bn_param)
		
		# runtime_depth
		self.input_stem_skipping = 0
		self.runtime_depth = [0] * len(n_block_list)

	def call(self, x):
		for layer in self.input_stem:
			if self.input_stem_skipping > 0 \
			and  isinstance(layer, ResidualBlock) and\
				 isinstance(layer.shortcut, IdentityLayer):
				pass
			else:
				x = layer(x)
		x = self.max_pooling(x)
		for stage_id, block_idx in enumerate(self.grouped_block_index):
			depth_param = self.runtime_depth[stage_id]
			active_idx = block_idx[:len(block_idx) - depth_param]
			for idx in active_idx:
				x = self.blocks[idx](x)
		x = self.global_avg_pool(x)
		x = self.classifier(x)

		return x
	
	@property
	def grouped_block_index(self):
		info_list = []
		block_index_list = []
		for i, block in enumerate(self.blocks):
			if not isinstance(block.downsample, IdentityLayer) and len(block_index_list) > 0:
				info_list.append(block_index_list)
				block_index_list = []
			block_index_list.append(i)
		if len(block_index_list) > 0:
			info_list.append(block_index_list)
		return info_list

	def set_active_subnet(self, d=None, e=None, w=None, **kwargs):
		depth = val2list(d, len(ResNets.BASE_DEPTH_LIST) + 1)
		expand_ratio = val2list(e, len(self.blocks))
		width_mult = val2list(w, len(ResNets.BASE_DEPTH_LIST) + 2)

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

	def sample_active_subnet(self):
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
		self.set_active_subnet(**arch_config)
		return arch_config
	
	def re_organize_middle_weights(self, expand_ratio_stage=0):
		print("re_organize_middle_weights")
		for block in self.blocks:
			block.re_organize_middle_weights(expand_ratio_stage)

	def extra_repr(self):
		 return (f'num_classes={self._num_classes}, '
				f'channel_last={self._channel_last}'
				f'width_mult={self._width_mult}'
				f'dropout_rate={self._dropout_rate}'
				f'settings={self._settings}')

	def summary(self):
		_str = ''
		for layer in self.input_stem:
			if self.input_stem_skipping > 0 \
					and isinstance(layer, ResidualBlock) and\
					    isinstance(layer.shortcut, IdentityLayer):
				pass
			else:
				_str += layer.module_str + '\n'
		_str += 'max_pooling(ks=3, stride=2)\n'
		for stage_id, block_idx in enumerate(self.grouped_block_index):
			depth_param = self.runtime_depth[stage_id]
			active_idx = block_idx[:len(block_idx) - depth_param]
			for idx in active_idx:
				#print(self.blocks[idx])
				_str += self.blocks[idx].module_str + '\n'
		_str += self.global_avg_pool.__repr__() + '\n'
		_str += self.classifier[0].module_str
		
		return _str
	
	def get_active_parameter_size(self):
		params = 0
		_str = ''
		for layer in self.input_stem:
			if self.input_stem_skipping > 0 \
					and isinstance(layer, ResidualBlock) and\
					    isinstance(layer.shortcut, IdentityLayer):
				pass
			else:
				param_num = layer.get_active_parameter_size
				params += param_num
				_str += layer.module_str + f', params:{param_num} ' + '\n'
		#_str += 'max_pooling(ks=3, stride=2)\n'
		for stage_id, block_idx in enumerate(self.grouped_block_index):
			depth_param = self.runtime_depth[stage_id]
			active_idx = block_idx[:len(block_idx) - depth_param]
			for idx in active_idx:
				param_num = self.blocks[idx].get_active_parameter_size
				params += param_num
				_str += self.blocks[idx].module_str + f', params:{param_num}' + '\n'
		#_str += self.global_avg_pool.__repr__() + '\n'
		param_num = self.classifier[0].get_active_parameter_size
		params += param_num
		_str += self.classifier[0].module_str + f', params:{param_num}'
		
		return params, _str

	def load_ofa_parameters(self, path, raise_if_missing=False):
		#with nn.parameter_scope('', OrderedDict()):
		#	nn.load_parameters(path)
		#	params = nn.get_parameters(grad_only=False)
		#self.set_ofa_parameters(params, raise_if_missing=raise_if_missing)
		super(SearchNetOFAMobileNetV3, self).load_ofa_parameters(path, raise_if_missing)
		
	"""def set_ofa_parameters(self, params, raise_if_missing=False):
		for prefix, module in self.get_modules():
			for name, p in module.parameters.items():
				key = prefix + ('/' if prefix else '') + name
				if new_key in params:
					pass
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
				nn.logger.info(f'`{new_key}` loaded.')"""

	def save_parameters(self, path=None, params=None, grad_only=False):
		# save the architectures
		# if isinstance(self._modified_resnet_features[3]._mixed, Mo.MixedOp):
		#     output_path = os.path.dirname(path)
		#     plot_resnet(self, os.path.join(output_path, 'arch'))

		super().save_parameters(path, params=params, grad_only=grad_only)

		base_path = path[0:path.rindex("/")]
		weight_str = path[path.rindex("/")+1:].split(".")[0]
		cur_epoch = weight_str.split("_")

class TrainNetOFAResNets(SearchNetOFAResNets):
	r"""ResNet Train Net"""
	def __init__(self, num_classes=1000, bn_param=(0.1, 1e-5), 
				dropout=0, depth_list=None, expand_ratio_list=None, 
				width_mult_list=None, genotype=None):
		super().__init__(num_classes=num_classes, bn_param=bn_param,
						 dropout=dropout, depth_list=depth_list,
						 expand_ratio_list=expand_ratio_list, 
						 width_mult_list=width_mult_list, )
		
		if genotype is not None:
			self.load_parameters(genotype)
			"""for _, module in self.get_modules():
				if isinstance(module, ChoiceBlock):
					idx = np.argmax(module._mixed._alpha.d)
					module._mixed = module._mixed._ops[idx]
		else:
			# pick random model
			for _, module in self.get_modules():
				if isinstance(module, DynamicResNetBottleneckBlock):
					idx = np.random.randint(len(module._mixed._alpha.d))
					module._mixed = module._mixed._ops[idx]"""
			  
	
			   