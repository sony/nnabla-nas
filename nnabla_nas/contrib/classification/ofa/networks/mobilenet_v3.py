from collections import OrderedDict
import copy

import nnabla as nn
import nnabla.functions as F

from ..modules import ResidualBlock, ConvLayer, LinearLayer, MBConvLayer, IdentityLayer
from ..ofa_modules.my_modules import MyGlobalAvgPool2d, MyNetwork
from ..ofa_modules.common_tools import make_divisible
from ..... import module as Mo


class MobileNetV3(MyNetwork):

    def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier):
        super(MobileNetV3, self).__init__()

        self.first_conv = first_conv
        self.blocks = Mo.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dims=True)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def call(self, input):
        x = self.first_conv(input)
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        x = self.global_avg_pool(x)
        x = self.feature_mix_layer(x)
        x = F.reshape(x, shape=(x.shape[0], -1))
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.final_expand_layer.module_str + '\n'
        _str += self.global_avg_pool.__repr__() + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @staticmethod
    def adjust_cfg(cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
        for i, (stage_id, block_config_list) in enumerate(cfg.items()):
            for block_config in block_config_list:
                if ks is not None and stage_id != '0':
                    block_config[0] = (ks, ks)
                if expand_ratio is not None and stage_id != '0':
                    block_config[-1] = expand_ratio
                    block_config[1] = None
                    if stage_width_list is not None:
                        block_config[2] = stage_width_list[i]
            if depth_param is not None and stage_id != '0':
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)]
                cfg[stage_id] = new_block_config_list
        return cfg

    @staticmethod
    def build_net_via_cfg(cfg, input_channel, last_channel, num_classes, dropout):
        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel=(3, 3), stride=(2, 2), use_bn=True, act_func='h_swish', ops_order='weight_bn_act'
        )
        # build mobile blocks
        feature_dim = input_channel
        blocks = []
        for stage_id, block_config_list in cfg.items():
            for k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio in block_config_list:
                mb_conv = MBConvLayer(
                    feature_dim, out_channel, k, stride, expand_ratio, mid_channel, act_func, use_se
                )
                if stride == (1, 1) and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mb_conv, shortcut))
                feature_dim = out_channel
        # final expand layer
        final_expand_layer = ConvLayer(
            feature_dim, feature_dim * 6, kernel=(1, 1), use_bn=True, act_func='h_swish', ops_order='weight_bn_act',
        )
        # feature mix layer
        feature_mix_layer = ConvLayer(
            feature_dim * 6, last_channel, kernel=(1, 1), with_bias=False, use_bn=False, act_func='h_swish',
        )
        # classifier
        classifier = LinearLayer(last_channel, num_classes, dropout=dropout)

        return first_conv, blocks, final_expand_layer, feature_mix_layer, classifier

    """def set_bn_param(self, decay_rate, eps, gn_channel_per_group=None, **kwargs):
        set_bn_param(self, decay_rate, eps, gn_channel_per_group, **kwargs)

    def get_bn_param(self):
        return get_bn_param(self)

    def loss(self, outputs, targets, loss_weights=None):
        return cross_entropy_loss_with_label_smoothing(outputs[0], targets[0])"""

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


class MobileNetV3Large(MobileNetV3):

    def __init__(self,
                 num_classes=1000,
                 bn_param=(0.9, 1e-5),
                 dropout=0.2,
                 width_mult=1.0,
                 ks=None,
                 expand_ratio=None,
                 depth_param=None,
                 stage_width_list=None,
                 ):

        input_channel = 16
        last_channel = 1280
        input_channel = make_divisible(input_channel * width_mult)
        last_channel = make_divisible(last_channel * width_mult) \
            if width_mult > 1.0 else last_channel

        cfg = {
            #    k,     exp,    c,      se,         nl,         s,      e,
            '0': [
                [(3, 3), 16, 16, False, 'relu', (1, 1), 1],
            ],
            '1': [
                [(3, 3), 64, 24, False, 'relu', (2, 2), None],  # 4
                [(3, 3), 72, 24, False, 'relu', (1, 1), None],  # 3
            ],
            '2': [
                [(5, 5), 72, 40, True, 'relu', (2, 2), None],  # 3
                [(5, 5), 120, 40, True, 'relu', (1, 1), None],  # 3
                [(5, 5), 120, 40, True, 'relu', (1, 1), None],  # 3
            ],
            '3': [
                [(3, 3), 240, 80, False, 'h_swish', (2, 2), None],  # 6
                [(3, 3), 200, 80, False, 'h_swish', (1, 1), None],  # 2.5
                [(3, 3), 184, 80, False, 'h_swish', (1, 1), None],  # 2.3
                [(3, 3), 184, 80, False, 'h_swish', (1, 1), None],  # 2.3
            ],
            '4': [
                [(3, 3), 480, 112, True, 'h_swish', (1, 1), None],  # 6
                [(3, 3), 672, 112, True, 'h_swish', (1, 1), None],  # 6
            ],
            '5': [
                [(5, 5), 672, 160, True, 'h_swish', (2, 2), None],  # 6
                [(5, 5), 960, 160, True, 'h_swish', (1, 1), None],  # 6
                [(5, 5), 960, 160, True, 'h_swish', (1, 1), None],  # 6
            ]
        }

        cfg = self.adjust_cfg(cfg, ks, expand_ratio, depth_param, stage_width_list)
        # width multiplier on mobile setting, change `exp: 1` and `c: 2`
        for stage_id, block_config_list in cfg.items():
            for block_config in block_config_list:
                if block_config[1] is not None:
                    block_config[1] = make_divisible(block_config[1] * width_mult)
                block_config[2] = make_divisible(block_config[2] * width_mult)
        first_conv, blocks, final_expand_layer, feature_mix_layer, classifier = self.build_net_via_cfg(
            cfg, input_channel, last_channel, num_classes, dropout
        )
        super(MobileNetV3Large, self).__init__(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        # set bn param
        self.set_bn_param(*bn_param)
