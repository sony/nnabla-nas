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

from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
from nnabla.utils.save import save
import numpy as np

from nnabla_nas.contrib.classification import misc
from nnabla_nas.contrib.classification.base import ClassificationModel as Model
import nnabla_nas.module as mo
from nnabla_nas.module import static as smo
from nnabla_nas.module.parameter import Parameter


class SepConv(misc.SepConv, smo.Module):
    r"""
    A static separable convolution

    Args:
        parents (list): a list of static modules that
            are parents to this module
        name (string, optional): the name of the module
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`, optional): Padding sizes for
            dimensions. Defaults to None.
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
        dilation (:obj:`tuple` of :obj:`int`, optional): Dilation sizes for
            dimensions. Defaults to None.
        group (int, optional): Number of groups of channels. This makes
            connections across channels more sparse by grouping connections
            along map direction. Defaults to 1.
        w_init (:obj:`nnabla.initializer.BaseInitializer`
            or :obj:`numpy.ndarray`, optional):
            Initializer for weight. By default, it is initialized with
            :obj:`nnabla.initializer.UniformInitializer` within the range
            determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.
        b_init (:obj:`nnabla.initializer.BaseInitializer`
            or :obj:`numpy.ndarray`, optional):
            Initializer for bias. By default, it is initialized with zeros if
            `with_bias` is `True`.
        base_axis (:obj:`int`, optional): Dimensions up to `base_axis` are
            treated as the sample dimensions. Defaults to 1.
        fix_parameters (bool, optional): When set to `True`, the weights and
            biases will not be updated. Defaults to `False`.
        rng (numpy.random.RandomState, optional): Random generator for
            Initializer.  Defaults to None.
        with_bias (bool, optional): Specify whether to include the bias term.
            Defaults to `True`.
        channel_last(bool, optional): If True, the last dimension is
            considered as channel dimension, a.k.a NHWC order. Defaults to
            `False`.
    """

    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        misc.SepConv.__init__(self, *args, **kwargs)
        smo.Module.__init__(self,
                            parents=parents,
                            name=name,
                            eival_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class SepConvBN(smo.Graph):
    """
    A static separable convolution that applies batchnorm and relu at the end.

    Args:
        parents (list): a list of static modules that
            are parents to this module
        name (string, optional): the name of the module
        out_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3,5).
        dilation (:obj:`tuple` of :obj:`int`, optional): Dilation sizes for
            dimensions. Defaults to None.
    """

    def __init__(self, parents, out_channels,
                 kernel, dilation,
                 name='', eval_prob=None):
        smo.Graph.__init__(self,
                           parents=parents,
                           name=name,
                           eval_prob=eval_prob)
        self._out_channels = out_channels

        if dilation is None:
            pad = tuple([ki//2 for ki in kernel])
        else:
            pad = tuple([(ki//2)*di for ki, di in zip(kernel, dilation)])

        self.append(SepConv(parents=parents,
                            name='{}/SepConv_1'.format(self.name),
                            in_channels=parents[0].shape[1],
                            out_channels=out_channels,
                            kernel=kernel, pad=pad,
                            dilation=dilation,
                            with_bias=False,
                            eval_prob=eval_prob))

        self.append(SepConv(parents=[self[-1]],
                            name='{}/SepConv_2'.format(self.name),
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel=kernel, pad=pad,
                            dilation=dilation,
                            with_bias=False,
                            eval_prob=eval_prob))

        self.append(smo.BatchNormalization(parents=[self[-1]],
                                           n_features=self._out_channels,
                                           name='{}/bn'.format(self.name),
                                           n_dims=4))
        self.append(smo.ReLU(parents=[self[-1]],
                             name='{}/relu'.format(self.name)))


class SepConv3x3(SepConvBN):
    """
    A static separable convolution of shape 3x3 that applies batchnorm and relu at the end.

    Args:
        parents (list): a list of static modules that
            are parents to this module
        name (string, optional): the name of the module
        channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
    """

    def __init__(self, parents, channels, name='', eval_prob=None):
        SepConvBN.__init__(self,
                           parents=parents,
                           out_channels=channels,
                           kernel=(3, 3),
                           dilation=None,
                           name=name,
                           eval_prob=eval_prob)


class SepConv5x5(SepConvBN):
    """
    A static separable convolution of shape 5x5 that applies batchnorm and relu at the end.

    Args:
        parents (list): a list of static modules that
            are parents to this module
        name (string, optional): the name of the module
        channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
    """

    def __init__(self, parents, channels, name='', eval_prob=None):
        SepConvBN.__init__(self,
                           parents=parents,
                           out_channels=channels,
                           kernel=(5, 5),
                           dilation=None,
                           name=name,
                           eval_prob=eval_prob)


class DilSepConv3x3(SepConvBN):
    """
    A static dilated separable convolution of shape 3x3 that applies batchnorm and relu at the end.

    Args:
        parents (list): a list of static modules that
            are parents to this module
        name (string, optional): the name of the module
        channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
    """

    def __init__(self, parents, channels, name='', eval_prob=None):
        SepConvBN.__init__(self,
                           parents=parents,
                           out_channels=channels,
                           kernel=(3, 3),
                           dilation=(2, 2),
                           name=name,
                           eval_prob=eval_prob)


class DilSepConv5x5(SepConvBN):
    """
    A static dilated separable convolution of shape 5x5 that applies batchnorm and relu at the end.

    Args:
        parents (list): a list of static modules that
            are parents to this module
        name (string, optional): the name of the module
        channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of output channels). For example, to apply
            convolution on an input with 16 types of filters, specify 16.
    """

    def __init__(self, parents, channels, name='', eval_prob=None):
        SepConvBN.__init__(self,
                           parents=parents,
                           out_channels=channels,
                           kernel=(5, 5),
                           dilation=(2, 2),
                           name=name,
                           eval_prob=eval_prob)


class MaxPool3x3(smo.MaxPool):
    """
    A static max pooling of size 3x3

    Args:
        parents (list): a list of static modules that
            are parents to this module
        name (string, optional): the name of the module
    """

    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        smo.MaxPool.__init__(self,
                             parents=parents,
                             kernel=(3, 3),
                             stride=(1, 1),
                             pad=(1, 1),
                             name=name,
                             eval_prob=eval_prob)
        self.bn = mo.BatchNormalization(n_features=self.parents[0].shape[1],
                                        n_dims=4)
        self.relu = mo.ReLU()

    def call(self, *inputs):
        return self.relu(self.bn(smo.MaxPool.call(self,
                                                  *inputs)))


class AveragePool3x3(smo.AvgPool):
    """
    A static avg pooling of size 3x3

    Args:
        parents (list): a list of static modules that
            are parents to this module
        name (string, optional): the name of the module
    """

    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        smo.AvgPool.__init__(self,
                             parents=parents,
                             kernel=(3, 3),
                             stride=(1, 1),
                             pad=(1, 1),
                             name=name,
                             eval_prob=eval_prob)
        self.bn = mo.BatchNormalization(
            n_features=self.parents[0].shape[1], n_dims=4)
        self.relu = mo.ReLU()

    def call(self, *inputs):
        return self.relu(self.bn(smo.AvgPool.call(self,
                                                  *inputs)))


ZOPH_CANDIDATES = [SepConv3x3,
                   SepConv5x5,
                   DilSepConv3x3,
                   DilSepConv5x5,
                   MaxPool3x3,
                   AveragePool3x3,
                   smo.Identity,
                   smo.Zero]


class ZophBlock(smo.Graph):
    """
    A zoph block as defined in [Bender et. al]

    Args:
        parents (list): a list of static modules that
            are parents to this module
        name (string, optional): the name of the module
        candidates (list): the candidate modules instantiated within this block (e.g. ZOPH_CANDIDATES)
        channels (int): the number of output channels of this block
        join_parameters (nnabla variable, optional): the architecture parameters used to join the outputs
            of the candidate modules. join_parameters must have the same number of elements as we have candidates.

    References:
        - Bender, Gabriel. "Understanding and simplifying one-shot architecture search." (2019).
    """

    def __init__(self, parents, candidates,
                 channels, name='', join_parameters=None):
        self._candidates = candidates
        self._channels = channels
        if join_parameters is None:
            self._join_parameters = Parameter(shape=(len(candidates),))
        else:
            self._join_parameters = join_parameters
        smo.Graph.__init__(self,
                           parents=parents,
                           name=name)

        join_prob = F.softmax(self._join_parameters)

        # add an input concatenation
        input_con = smo.Merging(name='{}/input_con'.format(self.name),
                                parents=self.parents,
                                mode='concat',
                                axis=1,
                                eval_prob=F.sum(join_prob[:-1]))
        self.append(input_con)
        input_conv = smo.Conv(name='{}/input_conv'.format(self.name),
                              parents=[input_con],
                              in_channels=input_con.shape[1],
                              out_channels=self._channels,
                              kernel=(1, 1),
                              eval_prob=F.sum(join_prob[:-1]))
        self.append(input_conv)
        self.append(smo.BatchNormalization(name='{}/input_conv_bn'.format(
                                           self.name),
                                           parents=[self[-1]],
                                           n_dims=4,
                                           n_features=self._channels))

        self.append(smo.ReLU(name='{}/input_conv/relu'.format(self.name),
                             parents=[self[-1]]))

        for i, ci in enumerate(self._candidates):
            self.append(ci(name='{}/candidate_{}'.format(self.name, i),
                           parents=[input_conv],
                           channels=self._channels,
                           eval_prob=join_prob[i]))
        self.append(smo.Join(name='{}/join'.format(self.name),
                             parents=self[4:],
                             join_parameters=self._join_parameters))


class ZophCell(smo.Graph):
    """
    A zoph cell that consists of multiple zoph blocks, as defined in [Bender et. al]

    Args:
        parents (list): a list of static modules that
            are parents to this module
        name (string, optional): the name of the module
        candidates (list): the candidate modules instantiated within this block (e.g. ZOPH_CANDIDATES)
        channels (int): the number of output channels of this block
        join_parameters (list of nnabla variable, optional): lift of the architecture
            parameters used to join the outputs
            of the candidate modules. each element in join_parameters must have
            the same number of elements as we have candidates.
            The length of this list must be n_modules.

    References:
        - Bender, Gabriel. "Understanding and simplifying one-shot architecture search." (2019).
    """

    def __init__(self, parents, candidates, channels, name='',
                 n_modules=3, reducing=False, join_parameters=[None]*3):
        self._candidates = candidates
        self._channels = channels
        self._n_modules = n_modules
        self._reducing = reducing
        self._join_parameters = join_parameters
        smo.Graph.__init__(self, parents=parents, name=name)

        # match the input dimensions
        shapes = [(list(ii.shape) + 4 * [1])[:4] for ii in self.parents]
        min_shape = np.min(np.array(shapes), axis=0)
        self._shape_adaptation = {i: np.array(si[2:]) / min_shape[2:]
                                  for i, si in enumerate(shapes)
                                  if tuple(si[2:]) != tuple(min_shape[2:])}

        # perform the input channel projection, using pointwise convolutions
        projected_inputs = []
        for i, ii in enumerate(self.parents):
            self.append(smo.Conv(name='{}/input_conv_{}'.format(self.name, i),
                                 parents=[ii], in_channels=ii.shape[1],
                                 out_channels=self._channels,
                                 kernel=(1, 1), with_bias=False))
            self.append(smo.BatchNormalization(name='{}/input_bn_{}'.format(
                                               self.name, i),
                                               parents=[self[-1]],
                                               n_dims=4,
                                               n_features=self._channels))
            self.append(smo.ReLU(name='{}/input_conv_{}/relu'.format(
                                 self.name, i),
                                 parents=[self[-1]]))
            projected_inputs.append(self[-1])

        # perform shape adaptation, using pooling, if needed
        for i, pii in enumerate(projected_inputs):
            if i in self._shape_adaptation:
                self.append(smo.MaxPool(name='{}/shape_adapt'
                                        '_pool_{}'.format(self.name, i),
                                        parents=[pii],
                                        kernel=self._shape_adaptation[i],
                                        stride=self._shape_adaptation[i]))
                projected_inputs[i] = self[-1]

        if self._reducing:
            for i, pii in enumerate(projected_inputs):
                self.append(smo.MaxPool(name='{}/reduce'
                                        '_pool_{}'.format(self.name, i),
                                        parents=[pii],
                                        kernel=(2, 2), stride=(2, 2)))
                projected_inputs[i] = self[-1]

        cell_modules = projected_inputs

        for i in range(self._n_modules):
            self.append(ZophBlock(name='{}/zoph'
                                  '_block_{}'.format(self.name, i),
                                  parents=cell_modules[:i+2],
                                  candidates=self._candidates,
                                  channels=self._channels,
                                  join_parameters=self._join_parameters[i]))
            cell_modules.append(self[-1])
        # perform output concatenation
        self.append(smo.Merging(name=self.name+'/output_concat',
                                parents=cell_modules, mode='concat'))


class SearchNet(Model, smo.Graph):
    """
    A search space as defined in [Bender et. al]

    Args:
        name (string, optional): the name of the module
        input_shape (tuple): the shape of the network input
        n_classes (int): the number of output classes
        stem_channels (int): the number of channels for the stem convolutions
        cells (list): the type of the cells used within this search space
        cell_depth (list): the number of modules within each cell
        reducing (list): specifies for each cell if it reduces the feature map dimensions through pooling
        join_parameters (list): the join_parameters used in each cell and block.
        candidates (list, optional): the candidate modules instantiated within this block (e.g. ZOPH_CANDIDATES)
        mode (string): the mode which the join modules within this network use

    References:
        - Bender, Gabriel. "Understanding and simplifying one-shot architecture search." (2019).
    """

    def __init__(self, name='', input_shape=(3, 32, 32),
                 n_classes=10, stem_channels=128,
                 cells=[ZophCell]*3, cell_depth=[7]*3,
                 cell_channels=[128, 256, 512],
                 reducing=[False, True, True],
                 join_parameters=[[None]*7]*3,
                 candidates=ZOPH_CANDIDATES, mode='sample'):
        smo.Graph.__init__(self, parents=[], name=name)
        self._n_classes = n_classes
        self._stem_channels = stem_channels
        self._cells = cells
        self._cell_depth = cell_depth
        self._cell_channels = cell_channels
        self._join_parameters = join_parameters
        self._reducing = reducing
        self._candidates = candidates
        self._input_shape = (1,) + input_shape
        self._input = smo.Input(
            name='{}/input'.format(self.name),
            value=nn.Variable(self._input_shape))
        self._mode = mode
        # 1. add the stem convolutions
        self.append(smo.Conv(name='{}/stem'
                             '_conv_1'.format(self.name),
                             parents=[self._input],
                             in_channels=self._input.shape[1],
                             out_channels=self._stem_channels,
                             kernel=(7, 7), pad=(3, 3)))
        self.append(smo.BatchNormalization(name='{}/stem_bn'.format(self.name),
                                           parents=[self[-1]],
                                           n_dims=4,
                                           n_features=self._stem_channels))
        self.append(smo.ReLU(name='{}/stem_relu'.format(self.name),
                             parents=[self[-1]]))
        self.append(smo.Conv(name='{}/stem'
                             '_conv_2'.format(self.name),
                             parents=[self[-1]],
                             in_channels=self._stem_channels,
                             out_channels=self._stem_channels,
                             kernel=(3, 3), pad=(1, 1)))
        self.append(smo.BatchNormalization(name='{}/stem2_bn'.format(
                                           self.name),
                                           parents=[self[-1]],
                                           n_dims=4,
                                           n_features=self._stem_channels))
        self.append(smo.ReLU(name='{}/stem2_relu'.format(self.name),
                             parents=[self[-1]]))
        # add the first 2 cells
        self.append(self._cells[0](name='{}/cell_{}'.format(self.name, 0),
                                   parents=[self[3], self[6]],
                                   candidates=self._candidates,
                                   n_modules=self._cell_depth[0],
                                   channels=self._cell_channels[0],
                                   join_parameters=self._join_parameters[0],
                                   reducing=self._reducing[0]))
        self.append(self._cells[1](name='{}/cell_{}'.format(self.name, 1),
                                   parents=[self[6], self[7]],
                                   candidates=self._candidates,
                                   n_modules=self._cell_depth[1],
                                   channels=self._cell_channels[1],
                                   join_parameters=self._join_parameters[1],
                                   reducing=self._reducing[1]))
        # 2. add the cells using shared architecture parameters
        for i, celli in enumerate(zip(self._cells[2:], self._cell_depth[2:],
                                      self._cell_channels[2:],
                                      self._join_parameters[2:],
                                      self._reducing[2:])):
            self.append(celli[0](name='{}/cell_{}'.format(self.name, i+2),
                                 parents=self[-2:],
                                 candidates=self._candidates,
                                 n_modules=celli[1],
                                 channels=celli[2],
                                 join_parameters=celli[3],
                                 reducing=celli[4]))

        # 3. add output convolutions and global average pooling layers
        self.append(smo.Conv(name='{}/output_conv_1'.format(self.name),
                             parents=[self[-1]],
                             in_channels=self[-1].shape[1],
                             out_channels=self._n_classes,
                             kernel=(1, 1)))
        self.append(smo.BatchNormalization(name='{}/output_bn'.format(
                                           self.name),
                                           parents=[self[-1]],
                                           n_dims=4,
                                           n_features=self._n_classes))
        self.append(smo.ReLU(name='{}/output_relu'.format(self.name),
                             parents=[self[-1]]))

        self.append(smo.GlobalAvgPool(
            name='{}/global_average_pool'.format(self.name),
            parents=[self[-1]]))
        self.append(smo.Collapse(name='{}/output_reshape'.format(self.name),
                                 parents=[self[-1]]))

        for mi in self.get_arch_modules():
            mi.mode = self._mode

    @property
    def modules_to_profile(self):
        return [smo.Identity,
                smo.Zero,
                smo.Conv,
                smo.Join,
                smo.ReLU,
                smo.BatchNormalization,
                smo.Merging,
                SepConv3x3,
                SepConv5x5,
                DilSepConv3x3,
                DilSepConv5x5,
                MaxPool3x3,
                AveragePool3x3,
                smo.MaxPool,
                smo.GlobalAvgPool,
                smo.Collapse]

    @property
    def input_shapes(self):
        return [self._input.shape]

    def get_arch_modules(self):
        ans = []
        for name, module in self.get_modules():
            if isinstance(module, smo.Join):
                ans.append(module)
        return ans

    def get_net_modules(self, active_only=False):
        ans = []
        for name, module in self.get_modules():
            if isinstance(module,
                          smo.Module) and not isinstance(module, smo.Join):
                if active_only:
                    if module._value is not None:
                        ans.append(module)
                    else:
                        pass
                else:
                    ans.append(module)
        return ans

    def get_net_parameters(self, grad_only=False):
        param = OrderedDict()
        for key, val in self.get_parameters(grad_only).items():
            if 'join' not in key:
                param[key] = val
        return param

    def get_arch_parameters(self, grad_only=False):
        param = OrderedDict()
        for key, val in self.get_parameters(grad_only).items():
            if 'join' in key:
                param[key] = val
        return param

    def get_latency(self, estimator, active_only=True):
        latencies = {}
        for mi in self.get_net_modules(active_only=active_only):
            if type(mi) in self.modules_to_profile:
                latencies[mi.name] = estimator.predict(mi)
        return latencies

    def __call__(self, input):
        self.reset_value()
        self._input._value = input
        return self._recursive_call()

    def summary(self):
        r"""Summary of the model."""
        str_summary = ''
        for mi in self.get_arch_modules():
            mi._sel_p.forward()
            str_summary += mi.name + "/"
            str_summary += mi.parents[np.argmax(mi._join_parameters.d)].name
            str_summary += "/" + str(np.max(mi._sel_p.d)) + "\n"

        str_summary += "Instantiated modules are:\n"
        for mi in self.get_net_modules(active_only=True):
            if isinstance(mi, smo.Module):
                try:
                    mi._eval_prob.forward()
                except Exception:
                    pass
                str_summary += mi.name + " chosen with probability "
                str_summary += str(mi._eval_prob.d) + "\n"
        return str_summary

    def save(self, output_path):
        gvg = self.get_gv_graph()
        gvg.render(output_path+'/graph')

    def save_modules_nnp(self, path, active_only=False):
        mods = self.get_net_modules(active_only=active_only)

        for mi in mods:
            if type(mi) in self.modules_to_profile:
                print(type(mi))
                inp = [nn.Variable((1,)+si[1:]) for si in mi.input_shapes]

                out = mi.call(*inp)
                filename = path + mi.name + '.nnp'
                d = {str(i): inpi for i, inpi in enumerate(inp)}
                contents = {'networks': [{'name': mi.name,
                                          'batch_size': 1,
                                          'outputs': {'out': out},
                                          'names': d}],
                            'executors': [{'name': 'runtime',
                                           'network': mi.name,
                                           'data': ['inp'],
                                           'output': ['out']}]}

                save(filename, contents)


class TrainNet(SearchNet):
    """
    A search space as defined in [Bender et. al]. Its the same as SearchNet, just that mode is fixed to 'max'.

    Args:
        name (string, optional): the name of the module
        input_shape (tuple): the shape of the network input
        n_classes (int): the number of output classes
        stem_channels (int): the number of channels for the stem convolutions
        cells (list): the type of the cells used within this search space
        cell_depth (list): the number of modules within each cell
        reducing (list): specifies for each cell if it reduces the feature map dimensions through pooling
        join_parameters (list): the join_parameters used in each cell and block.
        candidates (list, optional): the candidate modules instantiated within this block (e.g. ZOPH_CANDIDATES)
        mode (string): the mode which the join modules within this network use

    References:
        - Bender, Gabriel. "Understanding and simplifying one-shot architecture search." (2019).
    """

    def __init__(self, name, input_shape=(3, 32, 32),
                 n_classes=10, stem_channels=128,
                 cells=[ZophCell]*3, cell_depth=[7]*3,
                 cell_channels=[128, 256, 512],
                 reducing=[False, True, True],
                 join_parameters=[[None]*7]*3,
                 candidates=ZOPH_CANDIDATES,
                 param_path=None,
                 *args, **kwargs):
        SearchNet.__init__(self, name=name,
                           input_shape=input_shape,
                           n_classes=n_classes,
                           stem_channels=stem_channels,
                           cells=cells, cell_depth=cell_depth,
                           reducing=reducing,
                           join_parameters=join_parameters,
                           candidates=ZOPH_CANDIDATES,
                           mode='max')

        if param_path is not None:
            self.load_parameters(param_path)
