from collections import Counter, OrderedDict
import copy

import nnabla as nn
import nnabla.functions as F

from ...base import ClassificationModel as Model
from ..modules import ResidualBlock, ResNetBottleneckBlock, ConvLayer, LinearLayer, IdentityLayer
from ..ofa_modules.my_modules import MyGlobalAvgPool2d, MyNetwork
from ..ofa_modules.common_tools import make_divisible
from ..... import module as Mo


class ResNets(MyNetwork):

    BASE_DEPTH_LIST = [2, 2, 4, 2]
    STAGE_WIDTH_LIST = [256, 512, 1024, 2048]

    def __init__(self, input_stem, blocks, classifier):
        super(ResNets, self).__init__()

        self.input_stem = Mo.ModuleList(input_stem)
        self.max_pooling = Mo.MaxPool(kernel=(3, 3), stride=(2, 2), pad=(1, 1))
        self.blocks = Mo.ModuleList(blocks)
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dims=False)
        self.classifier = classifier

    def call(self, x):
        for layer in self.input_stem:
            x = layer(x)
        x = self.max_pooling(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    """@property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.final_expand_layer.module_str + '\n'
        _str += self.global_avg_pool.__repr__() + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str"""

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

    def load_ofa_parameters(self, path, raise_if_missing=False):
        with nn.parameter_scope('', OrderedDict()):
            nn.load_parameters(path)
            params = nn.get_parameters(grad_only=False)
        self.set_ofa_parameters(params, raise_if_missing=raise_if_missing)

    def set_ofa_parameters(self, params, raise_if_missing=False):
        for prefix, module in self.get_modules():
            for name, p in module.parameters.items():
                key = prefix + ('/' if prefix else '') + name
                if '_layer/' in key:
                    new_key = key.replace('_layer/', '')
                elif 'linearlayer/' in key:
                    new_key = key.replace('linearlayer/', 'linear/')
                else:
                    new_key = key
                if new_key in params:
                    pass
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


class ResNet50D(ResNets):
    def __init__(self, num_classes=1000, width_mult=1.0, bn_param=(0.9, 1e-5), dropout=0,
                 expand_ratio=None, depth_param=None):

        expand_ratio = 0.25 if expand_ratio is None else expand_ratio

        input_channel = make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        mid_input_channel = make_divisible(input_channel // 2, MyNetwork.CHANNEL_DIVISIBLE)
        stage_width_list = ResNets.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE)

        depth_list = [3, 4, 6, 3]
        if depth_param is not None:
            for i, depth in enumerate(ResNets.BASE_DEPTH_LIST):
                depth_list[i] = depth + depth_param

        stride_list = [1, 2, 2, 2]

        # build input stem
        input_stem = [
            ConvLayer(3, mid_input_channel, (3, 3), stride=(2, 2), use_bn=True, act_func='relu'),
            ResidualBlock(
                ConvLayer(mid_input_channel, mid_input_channel, (3, 3), stride=(1, 1), use_bn=True, act_func='relu'),
                IdentityLayer(mid_input_channel, mid_input_channel)
            ),
            ConvLayer(mid_input_channel, input_channel, (3, 3), stride=(1, 1), use_bn=True, act_func='relu'),
        ]
        # blocks
        blocks = []
        for d, width, s in zip(depth_list, stage_width_list, stride_list):
            for i in range(d):
                stride = (s, s) if i == 0 else (1, 1)
                bottleneck_block = ResNetBottleneckBlock(
                    input_channel, width, kernel=(3, 3), stride=stride, expand_ratio=expand_ratio,
                    act_func='relu', downsample_mode='avgpool_conv',
                )
                blocks.append(bottleneck_block)
                input_channel = width
        # classifier
        classifier = LinearLayer(input_channel, num_classes, dropout=dropout)

        super(ResNet50D, self).__init__(input_stem, blocks, classifier)

        # set bn param
        self.set_bn_param(*bn_param)
