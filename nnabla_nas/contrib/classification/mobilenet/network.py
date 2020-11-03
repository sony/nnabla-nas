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

from collections import Counter
from collections import OrderedDict
import os

import nnabla.functions as F
from nnabla.utils.save import save

import nnabla as nn
import numpy as np

from .... import module as Mo
from ..base import ClassificationModel as Model
from .helper import plot_mobilenet
from .modules import CANDIDATES
from .modules import ChoiceBlock
from .modules import ConvBNReLU
from .modules import InvertedResidual


def _make_divisible(x, divisible_by=8):
    r"""It ensures that all layers have a channel number that is divisible by
    divisible_by."""
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def label_smoothing_loss(pred, label, label_smoothing=0.1):
    loss = F.softmax_cross_entropy(pred, label)
    if label_smoothing <= 0:
        return loss
    return (1 - label_smoothing) * loss - label_smoothing \
        * F.mean(F.log_softmax(pred), axis=1, keepdims=True)


class SearchNet(Model):
    r"""MobileNet V2 search space.

    This implementation is based on the PyTorch implementation.

    Args:
        num_classes (int): Number of classes
        width_mult (float, optional): Width multiplier - adjusts number of
            channels in each layer by this amount
        settings (list, optional): Network structure. Defaults to None.
        drop_rate (float, optional): Drop rate used in Dropout. Defaults to 0.
        candidates (list of str, optional): A list of candicates. Defaults to
            None.
        skip_connect (bool, optional): Whether the skip connect is used.
            Defaults to `True`.

    References:
    [1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.C., 2018.
        Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings
        of the IEEE conference on computer vision and pattern recognition
        (pp. 4510-4520).
    """

    def __init__(self,
                 num_classes=1000,
                 width_mult=1,
                 settings=None,
                 drop_rate=0,
                 candidates=None,
                 mode='sample',
                 skip_connect=True):

        self._num_classes = num_classes
        self._width_mult = width_mult
        self._skip_connect = skip_connect
        self._arch_idx = None  # keeps current max arch
        round_nearest = 8

        in_channels = 32
        last_channel = 1280

        # building first layer
        in_channels = _make_divisible(in_channels * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult),
            round_nearest
        )
        features = [ConvBNReLU(3, in_channels, stride=(2, 2))]

        first_cell_width = _make_divisible(16 * width_mult, 8)
        features += [CANDIDATES['MB1 3x3'](
            in_channels, first_cell_width, 1)]
        in_channels = first_cell_width

        if settings is None:
            settings = [
                # c, n, s
                [24, 4, 2],
                [32, 4, 2],
                [64, 4, 2],
                [96, 4, 1],
                [160, 4, 2],
                [320, 1, 1]
            ]
        self._settings = settings
        if candidates is None:
            candidates = [
                "MB3 3x3",
                "MB6 3x3",
                "MB3 5x5",
                "MB6 5x5",
                "MB3 7x7",
                "MB6 7x7"
            ]
        self._candidates = candidates
        # building inverted residual blocks
        for c, n, s in settings:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                curr_candidates = candidates.copy()
                if stride == 1 and in_channels == output_channel \
                        and skip_connect:
                    curr_candidates.append('skip_connect')
                features.append(
                    ChoiceBlock(in_channels, output_channel,
                                stride=stride, mode=mode,
                                ops=curr_candidates)
                )
                in_channels = output_channel

        # building last several layers
        features.append(ConvBNReLU(in_channels, self.last_channel,
                                   kernel=(1, 1)))
        # make it nn.Sequential
        self._features = Mo.Sequential(*features)

        # building classifier
        self._classifier = Mo.Sequential(
            Mo.GlobalAvgPool(),
            Mo.Dropout(drop_rate),
            Mo.Linear(self.last_channel, num_classes),
        )

    @property
    def modules_to_profile(self):
        return [Mo.Sequential,
                ConvBNReLU,
                Mo.Conv,
                Mo.BatchNormalization,
                Mo.ReLU6,
                ChoiceBlock,
                InvertedResidual,
                Mo.ModuleList,
                Mo.Dropout,
                Mo.Identity,
                Mo.Linear,
                Mo.MixedOp,
                Mo.GlobalAvgPool
                ]


    def get_net_parameters(self, grad_only=False):
        r"""Returns an `OrderedDict` containing model parameters.

        Args:
            grad_only (bool, optional): If sets to `True`, then only parameters
                with `need_grad=True` are returned. Defaults to False.

        Returns:
            OrderedDict: A dictionary containing parameters.
        """
        p = self.get_parameters(grad_only)
        return OrderedDict([(k, v) for k, v in p.items() if 'alpha' not in k])

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

    def call(self, input):
        out = self._features(input)
        return self._classifier(out)

    def extra_repr(self):
        return (f'num_classes={self._num_classes}, '
                f'width_mult={self._width_mult}, '
                f'settings={self._settings}, '
                f'candidates={self._candidates}, '
                f'skip_connect={self._skip_connect}')

    def summary(self):
        def print_arch(arch_idx, op_names):
            str = 'NET SUMMARY:\n'
            for k, (c, n, s) in enumerate(self._settings):
                str += 'c={:<4} : '.format(c)
                for i in range(n):
                    idx = k*n+i
                    if (self._arch_idx is None or
                            arch_idx[idx] == self._arch_idx[idx]):
                        str += ' '
                    else:
                        str += '*'
                    str += '{:<30}; '.format(op_names[arch_idx[idx]])
                str += '\n'
            return str
        stats = []
        arch_params = self.get_arch_parameters()
        arch_idx = [np.argmax(m.d.flat) for m in arch_params.values()]
        count = Counter(arch_idx)
        op_names = self._candidates.copy()
        if self._skip_connect:
            op_names += ['skip_connect']
        txt = print_arch(arch_idx, op_names)
        total = len(arch_params)
        for k in range(len(op_names)):
            name = op_names[k]
            stats.append(name + f' = {count[k]/total*100:.2f}%\t')
        if self._arch_idx is not None:
            n_changes = sum(i != j for i, j in zip(arch_idx, self._arch_idx))
            txt += '\n Number of changes: {}({:.2f}%)\n'.format(
                n_changes, n_changes*100/len(arch_idx))
        self._arch_idx = arch_idx
        return txt + ''.join(stats)

    def save_parameters(self, path=None, params=None, grad_only=False):
        super().save_parameters(path, params=params, grad_only=grad_only)
        # save the architectures
        if isinstance(self._features[2]._mixed, Mo.MixedOp):
            output_path = os.path.dirname(path)
            plot_mobilenet(self, os.path.join(output_path, 'arch'))

    def loss(self, outputs, targets, loss_weights=None):
        assert len(outputs) == 1 and len(targets) == 1
        return F.mean(label_smoothing_loss(outputs[0], targets[0]))


    def get_net_modules(self, active_only=False):
        ans = []
        for name, module in self.get_modules():
            if isinstance(module, Mo.Module): #and not isinstance(module, smo.Join)
                ans.append(module)
        return ans

    def save_modules_nnp(self, path, active_only=False):
        """
            *** This script does not work  ***
            since the input shapes of the CANDIDATES are only defined at run time!

            Saves all modules of the network as individual nnp files,
            using folder structure given by name convention
            
            Args:
                path
                active_only: if True, only active modules are saved
        """
        mods = self.get_net_modules()
        idx = 0
        for mi in mods:
            if type(mi) in self.modules_to_profile:
                if len(mi.input_shapes) == 0:
                    print('NOT DEFINED: ', type(mi))
                    continue
                pass

                inp = [nn.Variable((1,)+si[1:]) for si in mi.input_shapes]
                out = mi.call(*inp)

                filename = path + mi.__module__ + '_' + str(idx) + '.nnp'
                pathname = os.path.dirname(filename)
                if not os.path.exists(pathname):
                    os.mkdir(pathname)

                d_dict = {str(i): inpi for i, inpi in enumerate(inp)}
                d_keys = [str(i) for i, inpi in enumerate(inp)]

                contents = {'networks': [{'name': mi.__module__,
                                          'batch_size': 1,
                                          'outputs': {'out': out},
                                          'names': d_dict}],
                            'executors': [{'name': 'runtime',
                                           'network': mi.__module__,
                                           'data': d_keys,
                                           'output': ['out']}]}
                
                save(filename, contents, variable_batch_size=False)
                idx = idx + 1

        #import pdb; pdb.set_trace()

    def save_net_nnp(self, path, inp, out, save_latency=False):
        """
            Saves whole net as one nnp
            Args:
                path
                inp: input of the created network
                out: output of the created network
                save_latency: calculate and save also latency
        """
        batch_size = inp.shape[0]

        name = '_whole_net'
        filename = path + name + '.nnp'
        pathname = os.path.dirname(filename)
        if not os.path.exists(pathname):
            os.mkdir(pathname)

        dict = {'0': inp}
        keys = ['0']

        contents = {'networks': [{'name': name,
                                  'batch_size': batch_size,
                                  'outputs': {'out': out},
                                  'names': dict}],
                    'executors': [{'name': 'runtime',
                                   'network': name,
                                   'data': keys,
                                   'output': ['out']}]}

        save(filename, contents, variable_batch_size=False)
        if save_latency:
            from nnabla_nas.utils.estimator import LatencyEstimator
            #estimation = LatencyEstimator(n_run = 100, ext_name='cuda', device_id = 0)
            estimation = LatencyEstimator(n_run = 100, ext_name='cpu')
            latency = estimation.get_estimation(self)
            filename = path + name + '.lat'
            with open(filename, 'w') as f:
                print(latency.__str__(), file=f)


    def convert_npp_to_onnx(self, path):
        """
            Finds all nnp files in the given path and its subfolders and converts them to ONNX
            For this to run smoothly, nnabla_cli must be installed and added to your python path.
            Args:
                path

        The actual bash shell command used is:
        > find <DIR> -name '*.nnp' -exec echo echo {} \| awk -F \\. \'\{print \"nnabla_cli convert -b 1 -d opset_11 \"\$0\" \"\$1\"\.\"\$2\"\.onnx\"\}\' \; | sh | sh
        which, for each file found with find, outputs the following:
        > echo <FILE>.nnp | awk -F \. '{print "nnabla_cli convert -b 1 -d opset_11 "$0" "$1"."$2".onnx"}'
        which, for each file, generates the final conversion command:
        > nnabla_cli convert -b 1 -d opset_11 <FILE>.nnp <FILE>.onnx

        """

        os.system('find ' + path + ' -name "*.nnp" -exec echo echo {} \|'
                  ' awk -F \\. \\\'{print \\\"nnabla_cli convert -b 1 -d opset_11 \\\"\$0\\\" \\\"\$1\\\"\.\\\"\$2\\\"\.onnx\\\"}\\\' \; | sh | sh'
                 )



class TrainNet(SearchNet):
    r"""MobileNet V2 Train Net.

    Args:
        num_classes (int): Number of classes
        width_mult (float, optional): Width multiplier - adjusts number of
            channels in each layer by this amount
        settings (list, optional): Network structure.
            Defaults to None.
        round_nearest (int, optional): Round the number of channels in
            each layer to be a multiple of this number. Set to 1 to turn
            off rounding.
        n_max (int, optional): The number of blocks. Defaults to 4.
        block: Module specifying inverted residual building block for
            mobilenet. Defaults to None.
        mode (str, optional): The sampling strategy ('full', 'max', 'sample').
            Defaults to 'full'.
        skip_connect (bool, optional): Whether the skip connect is used.
            Defaults to `True`.
        genotype(str, optional): The path to architecture file. Defaults to
            None.

    References:
    [1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.C., 2018.
        Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings
        of the IEEE conference on computer vision and pattern recognition
        (pp. 4510-4520).
    """

    def __init__(self,
                 num_classes=1000,
                 width_mult=1,
                 settings=None,
                 drop_rate=0,
                 candidates=None,
                 mode='sample',
                 skip_connect=True,
                 genotype=None):

        super().__init__(num_classes=num_classes, width_mult=width_mult,
                         settings=settings, drop_rate=drop_rate,
                         candidates=candidates, mode=mode,
                         skip_connect=skip_connect)

        if genotype is not None:
            self.load_parameters(genotype)
            for _, module in self.get_modules():
                if isinstance(module, ChoiceBlock):
                    idx = np.argmax(module._mixed._alpha.d)
                    module._mixed = module._mixed._ops[idx]
        else:
            # pick random model
            for _, module in self.get_modules():
                if isinstance(module, ChoiceBlock):
                    idx = np.random.randint(len(module._mixed._alpha.d))
                    module._mixed = module._mixed._ops[idx]
