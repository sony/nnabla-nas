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

import os

from collections import OrderedDict
from copy import deepcopy

import networkx as nx
import nnabla as nn
from nnabla.utils.save import save

import numpy as np

from nnabla_nas.module import static as smo


class RandomModule(smo.Graph):
    """
    A module that automatically aggregates all the output tensors generated by
    its parents. Therefore, we automatically adjusts the input channel count
    and the input feature map dimensions of each
    input through 1x1 convolution and pooling. The result is summed up.
    Please refer to [Xie et. al]

    Args:
        parents (list): the parent modules to this module
        name (string, optional): the name of the module
        channels (int): the number of output channels of this module

    References:
        - Xie, Saining, et al. "Exploring randomly wired neural
          networks for image recognition."  Proceedings of the IEEE
          International Conference on Computer Vision. 2019.
    """

    def __init__(self, parents, channels, name=''):
        smo.Graph.__init__(self,
                           parents=parents,
                           name=name)
        self._channels = channels
        shapes = [(list(ii.shape) + 4 * [1])[:4] for ii in self.parents]
        min_shape = np.min(np.array(shapes), axis=0)
        self._shape_adaptation = {i: np.array(si[2:]) / min_shape[2:]
                                  for i, si in enumerate(shapes)
                                  if tuple(si[2:]) != tuple(min_shape[2:])}
        projected_inputs = []

        # add an input convolution to project to the correct #channels
        for i, pi in enumerate(self.parents):
            self.append(smo.Conv(name='{}/input_conv_{}'.format(self.name, i),
                                 parents=[pi],
                                 in_channels=pi.shape[1],
                                 out_channels=self._channels,
                                 kernel=(1, 1)))
            self.append(
                smo.BatchNormalization(name='{}/input_conv_bn_{}'.format(
                    self.name, i),
                    parents=[self[-1]],
                    n_dims=4,
                    n_features=self._channels))
            self.append(
                smo.ReLU(name='{}/input_conv_relu_{}'.format(self.name, i),
                         parents=[self[-1]]))

            projected_inputs.append(self[-1])

        for i, pii in enumerate(projected_inputs):
            if i in self._shape_adaptation:
                self.append(smo.MaxPool(name='{}/shape_adapt'
                                        '_pool_{}'.format(self.name, i),
                                        parents=[pii],
                                        kernel=self._shape_adaptation[i],
                                        stride=self._shape_adaptation[i]))
                projected_inputs[i] = self[-1]
        if len(projected_inputs) > 1:
            self.append(smo.Merging(parents=projected_inputs,
                                    name='{}/merging'.format(self.name),
                                    mode='add'))


class Conv(RandomModule):
    """
    A convolution that accepts multiple parents. This convolution
    is a random module, meaning that it automatically adjusts the
    dimensions of all input tensors and aggregates the
    result before applying the convolution.

    Args:
        parents (list): the parent modules to this module
        name (string, optional): the name of the module
        channels (int): the number of output channels of this module
        kernel (tuple): the kernel shape
        pad (tuple): the padding scheme used
    """

    def __init__(self,
                 parents,
                 channels,
                 kernel,
                 pad,
                 name=''):
        RandomModule.__init__(self,
                              parents=parents,
                              channels=channels,
                              name=name)
        self._channels = channels
        self._kernel = kernel
        self._pad = pad
        self.append(smo.Conv(name='{}/conv'.format(self.name),
                             parents=[self[-1]],
                             in_channels=self[-1].shape[1],
                             out_channels=self._channels,
                             kernel=self._kernel,
                             pad=self._pad))
        self.append(smo.BatchNormalization(name='{}/conv_bn'.format(
                                           self.name),
                                           parents=[self[-1]],
                                           n_dims=4,
                                           n_features=self._channels))
        self.append(smo.ReLU(name='{}/conv_relu'.format(self.name),
                             parents=[self[-1]]))


class SepConv(RandomModule):
    """
    A separable convolution that accepts multiple parents. This convolution
    is a random module, meaning that it automatically adjusts the dimensions of
    all input tensors and aggregates the result before applying the
    convolution.

    Args:
        parents (list): the parent modules to this module
        name (string, optional): the name of the module
        channels (int): the number of output channels of this module
        kernel (tuple): the kernel shape
        pad (tuple): the padding scheme used
    """

    def __init__(self,
                 parents,
                 channels,
                 kernel,
                 pad,
                 name=''):
        RandomModule.__init__(self,
                              parents=parents,
                              channels=channels,
                              name=name)
        self._channels = channels
        self._kernel = kernel
        self._pad = pad
        self.append(smo.Conv(name='{}/conv_dw'.format(self.name),
                             parents=[self[-1]],
                             in_channels=self[-1].shape[1],
                             out_channels=self[-1].shape[1],
                             kernel=self._kernel,
                             group=1,
                             pad=self._pad))
        self.append(smo.Conv(name='{}/conv_pw'.format(self.name),
                             parents=[self[-1]],
                             in_channels=self[-1].shape[1],
                             out_channels=self._channels,
                             kernel=(1, 1)))
        self.append(smo.BatchNormalization(name='{}/conv_bn'.format(
                                           self.name),
                                           parents=[self[-1]],
                                           n_dims=4,
                                           n_features=self._channels))
        self.append(smo.ReLU(name='{}/conv_relu'.format(self.name),
                             parents=[self[-1]]))


class Conv3x3(Conv):
    """
    A convolution of shape 3x3 that accepts multiple parents. This convolution
    is a random module, meaning that it automatically adjusts the dimensions of
    all input tensors and aggregates the result before applying the
    convolution.

    Args:
        parents (list): the parent modules to this module
        name (string, optional): the name of the module
        channels (int): the number of output channels of this module
    """

    def __init__(self,
                 parents,
                 channels,
                 name=''):
        Conv.__init__(self,
                      parents=parents,
                      channels=channels,
                      name=name,
                      kernel=(3, 3),
                      pad=(1, 1))


class SepConv3x3(SepConv):
    """
    A separable convolution of shape 3x3 that accepts multiple parents.
    This convolution is a random module, meaning that
    it automatically adjusts the dimensions of all input tensors and aggregates
    the result before applying the
    convolution.

    Args:
        parents (list): the parent modules to this module
        name (string, optional): the name of the module
        channels (int): the number of output channels of this module
    """

    def __init__(self,
                 parents,
                 channels,
                 name=''):
        SepConv.__init__(self,
                         parents=parents,
                         channels=channels,
                         name=name,
                         kernel=(3, 3),
                         pad=(1, 1))


class Conv5x5(Conv):
    """
    A convolution of shape 5x5 that accepts multiple parents. This convolution
    is a random module, meaning that it automatically adjusts the dimensions
    of all input tensors and aggregates the result before applying the
    convolution.

    Args:
        parents (list): the parent modules to this module
        name (string, optional): the name of the module
        channels (int): the number of output channels of this module
    """

    def __init__(self,
                 parents,
                 channels,
                 name=''):
        Conv.__init__(self,
                      parents=parents,
                      channels=channels,
                      name=name,
                      kernel=(5, 5),
                      pad=(2, 2))


class SepConv5x5(SepConv):
    """
    A separable convolution of shape 5x5 that accepts multiple parents.
    This convolution is a random module, meaning that
    it automatically adjusts the dimensions of all input tensors and
    aggregates the result before applying the
    convolution.

    Args:
        parents (list): the parent modules to this module
        name (string, optional): the name of the module
        channels (int): the number of output channels of this module
    """

    def __init__(self,
                 parents,
                 channels,
                 name=''):
        SepConv.__init__(self,
                         parents=parents,
                         channels=channels,
                         name=name,
                         kernel=(5, 5),
                         pad=(2, 2))


class MaxPool2x2(RandomModule):
    """
    A max pooling module that accepts multiple parents. This pooling
    module is a random module, meaning that
    it automatically adjusts the dimensions of all input tensors and
    aggregates the result before applying the
    convolution.

    Args:
        parents (list): the parent modules to this module
        name (string, optional): the name of the module
        channels (int): ignored
    """

    def __init__(self,
                 parents,
                 channels,
                 name=''):
        RandomModule.__init__(self,
                              parents=parents,
                              channels=channels,
                              name=name)
        self.append(smo.MaxPool(parents=[self[-1]],
                                kernel=(2, 2),
                                stride=(2, 2),
                                name='{}/max_pool_2x2'.format(self.name)))


class AvgPool2x2(RandomModule):
    """
    A avg pooling module that accepts multiple parents. This pooling
    module is a random module, meaning that
    it automatically adjusts the dimensions of all input tensors and
    aggregates the result before applying the
    convolution.

    Args:
        parents (list): the parent modules to this module
        name (string, optional): the name of the module
        channels (int): ignored
    """

    def __init__(self,
                 parents,
                 channels,
                 name=''):
        RandomModule.__init__(self,
                              parents=parents,
                              channels=channels,
                              name=name)
        self.append(smo.AvgPool(parents=[self[-1]],
                                kernel=(2, 2),
                                stride=(2, 2),
                                name='{}/avg_pool_2x2'.format(self.name)))


RANDOM_CANDIDATES = [RandomModule,
                     SepConv3x3,
                     SepConv5x5,
                     RandomModule,
                     SepConv3x3,
                     SepConv5x5,
                     RandomModule,
                     SepConv3x3,
                     SepConv5x5,
                     MaxPool2x2,
                     AvgPool2x2]


class TrainNet(smo.Graph):
    """
    A randomly wired DNN that uses the Watts-Strogatz process to generate
    random DNN architectures. Please refer to [Xie et. al]

    Args:
        n_vertice (int): the number of random modules within this network
        input_shape (tuple): the shape of the input of this network
        n_classes (int): the number of output classes of this network
        candidates (list): a list of random_modules which are randomly
                            instantiated as vertices
        min_channels (int): the minimum channel count of a vertice
        max_channels (int): the maximum channel count of a vertice
        k (int): the connectivity parameter of the Watts-Strogatz process
        p (float): the re-wiring probability parameter of the
                Watts-Strogatz process
        name (string): the name of the network

    References:
        - Xie, Saining, et al. "Exploring randomly wired neural networks
             for image recognition." Proceedings of the IEEE International
             Conference on Computer Vision. 2019.
    """

    def __init__(self, n_vertices=20, input_shape=(3, 32, 32),
                 n_classes=10, candidates=RANDOM_CANDIDATES, min_channels=128,
                 max_channels=1024, k=4, p=0.75, name=''):
        smo.Graph.__init__(self,
                           parents=[],
                           name=name)
        self._input_shape = (1,) + input_shape
        self._n_vertices = n_vertices
        self._candidates = candidates
        self._n_classes = n_classes
        self._min_channels = min_channels
        self._max_channels = max_channels
        self._k = k
        self._p = p

        # 1. draw a random network graph
        g = self._get_random_graph(n_vertices,
                                   self._input_shape[1],
                                   output_channels=self._n_classes,
                                   candidates=self._candidates,
                                   min_channels=self._min_channels,
                                   max_channels=self._max_channels,
                                   k=self._k,
                                   p=self._p)

        self._init_modules_from_graph(g)

    def _init_modules_from_graph(self, graph):
        adj_matrix = nx.adjacency_matrix(graph).todense()
        sorted_nodes = np.argsort(graph.nodes)
        for i, ii in enumerate(sorted_nodes):
            p_idxs = np.where(np.ravel(adj_matrix[sorted_nodes, ii]) > 0)[0]
            if len(p_idxs) == 0:
                self.append(smo.Input(name='{}/input'.format(self.name),
                                      value=nn.Variable(self._input_shape)))
            else:
                rnd_class = self._candidates[
                    np.random.randint(0, len(self._candidates), 1)[0]]
                rnd_channels = np.random.randint(self._min_channels,
                                                 self._max_channels,
                                                 1)[0]
                parents = [self[pi] for pi in p_idxs]

                self.append(rnd_class(name='{}/{}'.format(self.name, i),
                                      parents=parents,
                                      channels=rnd_channels))

        self.append(smo.GlobalAvgPool(
            name='{}/global_average_pool'.format(self.name),
            parents=[self[-1]]))
        self.append(smo.Collapse(name='{}/output_reshape'.format(self.name),
                                 parents=[self[-1]]))

    def _get_random_graph(self,
                          n_vertices,
                          input_channels,
                          output_channels,
                          candidates=[],
                          min_channels=32,
                          max_channels=512,
                          k=10,
                          p=0.5):

        graph = nx.watts_strogatz_graph(n_vertices, k=k, p=p)

        # 1. make the graph directed, such that it is not cyclic
        G = nx.DiGraph()
        G.name = graph.name
        G.add_nodes_from(graph)
        G.add_edges_from(((u, v, deepcopy(data))
                          for u, nbrs in graph.adjacency()
                          for v, data in nbrs.items()
                          if v > u))
        G.graph = deepcopy(graph.graph)

        # 2. add a single input and output to the network
        adj_matrix = nx.adjacency_matrix(G).todense()
        inputs = np.where(np.ravel(np.sum(adj_matrix, axis=0) == 0))
        outputs = np.where(np.ravel(np.sum(adj_matrix, axis=1) == 0))
        G.add_node(-1)  # input
        G.add_node(n_vertices)  # output
        for i in inputs[0]:
            G.add_edge(-1, i)
        for o in outputs[0]:
            G.add_edge(o, n_vertices)
        return G

    @property
    def input_shapes(self):
        return [self[0].shape]

    @property
    def modules_to_profile(self):
        return [smo.ReLU,
                smo.BatchNormalization,
                smo.Join,
                smo.Merging,
                smo.Collapse,
                smo.Conv,
                smo.MaxPool,
                smo.AvgPool,
                smo.GlobalAvgPool,
                ]

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
        self[0]._value = input
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

    def save_graph(self, path):
        """
            save whole network/graph (in a PDF file)
            Args:
                path
        """
        gvg = self.get_gv_graph()
        gvg.render(path + '/graph')

    def save_net_nnp(self, path, inp, out, calc_latency=False,
                     func_real_latency=None, func_accum_latency=None):
        """
            Saves whole net as one nnp
            Args:
                path
                inp: input of the created network
                out: output of the created network
                calc_latency: flag for calc latency
                func_real_latency: function to use to calc actual latency
                func_accum_latency: function to use to calc accum. latency
                        this is, dissecting the network layer by layer,
                        calc. latency for each layer and add up the results

        """
        batch_size = inp.shape[0]

        name = self.name

        filename = path + name + '.nnp'
        pathname = os.path.dirname(filename)
        upper_pathname = os.path.dirname(pathname)
        if not os.path.exists(upper_pathname):
            os.mkdir(upper_pathname)
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

        if calc_latency:
            acc_latency = func_accum_latency.get_estimation(out)
            filename = path + name + '.acclat'
            with open(filename, 'w') as f:
                print(acc_latency.__str__(), file=f)

            func_real_latency.run()
            real_latency = float(func_real_latency.result['forward_all'])
            filename = path + name + '.realat'
            with open(filename, 'w') as f:
                print(real_latency.__str__(), file=f)

    def save_modules_nnp(self, path, active_only=False,
                         calc_latency=False, func_latency=None):
        """
            Saves all modules of the network as individual nnp files,
            using folder structure given by name convention
            Args:
                path
                active_only: if True, only active modules are saved
                calc_latency: flag for calc latency
                func_latency: function to use to calc latency of
                              each of the extracted modules
        """
        mods = self.get_net_modules(active_only=active_only)

        for mi in mods:
            if type(mi) in self.modules_to_profile:

                inp = [nn.Variable((1,)+si[1:]) for si in mi.input_shapes]
                out = mi.call(*inp)

                filename = path + mi.name + '.nnp'
                pathname = os.path.dirname(filename)
                upper_pathname = os.path.dirname(pathname)
                if not os.path.exists(upper_pathname):
                    os.mkdir(upper_pathname)
                if not os.path.exists(pathname):
                    os.mkdir(pathname)

                d_dict = {str(i): inpi for i, inpi in enumerate(inp)}
                d_keys = [str(i) for i, inpi in enumerate(inp)]

                contents = {'networks': [{'name': mi.name,
                                          'batch_size': 1,
                                          'outputs': {'out': out},
                                          'names': d_dict}],
                            'executors': [{'name': 'runtime',
                                           'network': mi.name,
                                           'data': d_keys,
                                           'output': ['out']}]}

                save(filename, contents, variable_batch_size=False)

                if calc_latency:
                    latency = func_latency.get_estimation(out)
                    filename = path + mi.name + '.acclat'
                    with open(filename, 'w') as f:
                        print(latency.__str__(), file=f)

    def convert_npp_to_onnx(self, path):
        """
            Finds all nnp files in the given path and its subfolders
            and converts them to ONNX
            For this to run smoothly, nnabla_cli must be installed
            and added to your python path.
            Args:
                path

        The actual bash shell command used is:
        > find <DIR> -name '*.nnp' -exec echo echo {} \| awk -F \\. \'\{print \"nnabla_cli convert -b 1 -d opset_11 \"\$0\" \"\$1\"\.\"\$2\"\.onnx\"\}\' \; | sh | sh  # noqa: E501,W605
        which, for each file found with find, outputs the following:
        > echo <FILE>.nnp | awk -F \. '{print "nnabla_cli convert -b 1 -d opset_11 "$0" "$1"."$2".onnx"}'  # noqa: E501,W605
        which, for each file, generates the final conversion command:
        > nnabla_cli convert -b 1 -d opset_11 <FILE>.nnp <FILE>.onnx

        """

        os.system('find ' + path + ' -name "*.nnp" -exec echo echo {} \|'  # noqa: E501,W605
                  ' awk -F \\. \\\'{print \\\"nnabla_cli convert -b 1 -d opset_11 \\\"\$0\\\" \\\"\$1\\\"\.\\\"\$2\\\"\.onnx\\\"}\\\' \; | sh | sh'  # noqa: E501,W605
                  )


if __name__ == '__main__':
    input_1 = smo.Input(name='input_1', value=nn.Variable((10, 16, 32, 32)))
    input_2 = smo.Input(name='input_2', value=nn.Variable((10, 32, 16, 16)))

    conv = Conv(name='test_conv',
                parents=[input_1, input_2],
                channels=64,
                kernel=(3, 3),
                pad=(1, 1))
    c3x3 = Conv3x3(name='test_c3x3',
                   parents=[input_1, input_2],
                   channels=64)
    c5x5 = Conv5x5(name='test_c5x5',
                   parents=[input_1, input_2],
                   channels=64)
    mp3x3 = MaxPool2x2(name='test_mp3x3',
                       parents=[input_1, input_2],
                       channels=64)
    ap3x3 = AvgPool2x2(name='test_ap3x3',
                       parents=[input_1, input_2],
                       channels=64)
    net = TrainNet(name='test_net')

    net.reset_value()
    out = net(nn.Variable((10, 3, 32, 32)))
    gvg = net.get_gv_graph(active_only=True)
    gvg.render('test_random')
