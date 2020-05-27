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

"""
This module defines static modules, i.e., modules that are aware
of the graph structure. This module defines static versions of
all dynamic modules defined in nnabla_nas.modules
"""
import operator

import nnabla as nn
import nnabla.functions as F
import numpy as np

import nnabla_nas.module as mo

from graphviz import Digraph


def _get_abs_string_index(obj, idx):
    """Get the absolute index for the list of modules"""
    idx = operator.index(idx)
    if not (-len(obj) <= idx < len(obj)):
        raise IndexError('index {} is out of range'.format(idx))
    if idx < 0:
        idx += len(obj)
    return str(idx)


class Module(mo.Module):
    r"""
    A static module is a module that encodes the graph structure, i.e.,
    it has parents and children. Static modules can be used to define
    graphs that can run run simple graph optimizations when
    constructing the nnabla graph.

    Args:
        parents (list): a list of static modules that
            are parents to this module
        name (string, optional): the name of the module
        eval_prob (nnabla variable, optional): the evaluation probability
            of this module

    Examples:
        >>> from nnabla_nas.module import static as smo
        >>> class MyModule(smo.Module):
        >>>     def __init__(self, parents):
        >>>         smo.Module.__init__(self, parents=parents)
        >>>         smo.Module.__init__(self, parents=parents)
        >>>         self.linear = mo.Linear(in_features=5, out_features=3)
        >>>
        >>>     def call(self, *input):
        >>>         return self.linear(*input)
        >>>
        >>> module_1 = smo.Module(name='module_1')
        >>> module_2 = smo.MyModule(parents=[module_1], name='module_2')
    """

    def __init__(self, parents=[], name='', eval_prob=None, *args, **kwargs):
        parent_type_mismatch = [not isinstance(pi, Module) for pi in parents]
        if sum(parent_type_mismatch) == 0:
            self._parents = parents
        else:
            raise RuntimeError

        for pi in parents:
            pi.add_child(self)
        self._children = []
        self._name = name
        self._value = None
        self._eval_probs = None
        self._shape = -1

        if eval_prob is None:
            self._eval_prob = nn.Variable.from_numpy_array(np.array(1.0))
        else:
            self._eval_prob = eval_prob
        mo.Module.__init__(self)

    def add_child(self, child):
        r"""
        Adds a static_module as a child to self

        Args:
            child (static_module): the module to add as a child
        """
        self._children.append(child)

    def _shape_function(self):
        r"""
        Calculates the output shape of this static_module.

        Returns:
            tuple: the shape of the output tensor
        """
        inputs = [nn.Variable(pi.shape) for pi in self.parents]
        dummy_graph = self.call(*inputs)
        return dummy_graph.shape

    @property
    def shape(self):
        r"""
        The output shape of the static_module.

        Returns:
            tuple: the shape of the output tensor
        """
        if self._shape == -1:
            self._shape = self._shape_function()
        return self._shape

    @property
    def input_shapes(self):
        r"""
        A list of input shapes of this module, i.e.,
        the output shapes of all parent modules.

        Returns:
            list: a list of tuples storing the
                output shape of all parent modules
        """
        return [pi.shape for pi in self._parents]

    @property
    def name(self):
        r"""
        The name of the module.

        Returns:
            string: the name of the module
        """
        return self._name

    @property
    def parents(self):
        r"""
        The parents of the module

        Returns:
            list: the parents of the module
        """
        return self._parents

    @property
    def children(self):
        r"""
        The child modules

        Returns:
            list: the children of the module
        """
        return self._children

    def call(self, *inputs):
        r"""
        The input to output mapping of the module.
        Given some inputs, it constructs
        the computational graph of this module. This method
        must be implemented for custom modules.

        Args:
            *input: the output of the parents

        Returns:
            nnabla variable: the output of the module

        Examples:
            >>> out = my_module(inp_a, inp_b)
        """
        raise NotImplementedError

    def _recursive_call(self):
        r"""
        Execute self.call on the output of all parent modules.

        Returns:
            nnabla variable: the output of the module
        """
        if self._value is None:
            self._value = self.call(*[pi() for pi in self.parents])
            self.need_grad = True
        return self._value

    def __call__(self):
        r"""
        Execute self.call on the output of all parent modules.

        Returns:
            nnabla variable: the output of the module
        """
        return self._recursive_call()

    @property
    def eval_prob(self):
        r"""
        The evaluation probability of this module. It is
        1.0 if not specified otherwise.

        Returns:
            nnabla variable: the evaluation probability
        """
        return self._eval_prob

    @eval_prob.setter
    def eval_prob(self, value):
        self._eval_prob = value

    @property
    def output(self):
        r"""
        The output module of this module. If the module is not
        a graph, it will return self.

        Returns:
            Module: the output module
        """
        return self

    def reset_value(self):
        r"""
        Resets all self._value, self.need_grad flags and self.shapes
        """
        self._value = None
        self.apply(need_grad=False)
        self._shape = -1


class Input(Module):
    r"""
    A static module that can serve as an input, i.e., it has no parents
    but is provided with a value which it can pass to its children.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
        value (nnabla variable): the nnabla variable which serves as
            the input value

    Examples:
       >>> import nnabla as nn
       >>> from nnabla_nas.module import static as smo
       >>> input = nn.Variable((10, 3, 32, 32))
       >>> inp_module = smo.Input(value=input)
    """

    def __init__(self, value=None, name='', eval_prob=None, *args, **kwargs):
        Module.__init__(self, name=name, parent=None, eval_prob=eval_prob)
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self.reset_value()
        self._value = v

    def call(self, *inputs):
        r"""
        The input module returns the plain input variable.
        """
        return self._value

    def _recursive_call(self):
        r"""
        Input module do not call anny parents.
        """
        self.need_grad = True
        return self.call(None)

    def _shape_function(self):
        return self._value.shape

    def reset_value(self):
        r"""
        the input module does not reset its value
        """
        self._shape = -1


class Identity(mo.Identity, Module):
    r"""
    The Identity module does not alter the input.
    It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module

    Examples:
       >>> import nnabla as nn
       >>> from nnabla_nas.module import static as smo
       >>>
       >>> nn.Variable((10, 3, 32, 32))
       >>>
       >>> inp_module = smo.Input(value=input)
       >>> identity = smo.Identity(parents=[inp_module])
    """
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        Module.__init__(self, parents, name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class Zero(mo.Zero, Module):
    r"""
    The Zero module returns a tensor with zeros, which has the
    same shape as the ouput of its parent. It accepts only
    a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module

    Examples:
        >>> my_module = Zero(parents=[...], name='my_module')
    """
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.Zero.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name, eval_prob=eval_prob)
        self._value = nn.Variable.from_numpy_array(
            np.zeros(self._parents[0].shape))
        if len(self._parents) > 1:
            raise RuntimeError

    def call(self, *inputs):
        self._value = nn.Variable.from_numpy_array(
            np.zeros(self._parents[0].shape))
        return self._value

    def _shape_function(self):
        return self._parents[0].shape

    def _recursive_call(self):
        self.need_grad = True
        return self.call(None)


class Conv(mo.Conv, Module):
    r"""
    The Conv module performs a convolution on the
    output of its parent. It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
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
        mo.Conv.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name,  eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class Linear(mo.Linear, Module):
    r"""
    The Linear module performs an affine transformation on the
    output of its parent. It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
        in_features (int): The size of each input sample.
        in_features (int): The size of each output sample.
        base_axis (int, optional): Dimensions up to `base_axis` are treated as
            the sample dimensions. Defaults to 1.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or
            :obj:`numpy.ndarray`): Initializer for weight. By default, it is
            initialized with :obj:`nnabla.initializer.UniformInitializer`
            within the range determined by
            :obj:`nnabla.initializer.calc_uniform_lim_glorot`.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or
            :obj:`numpy.ndarray`): Initializer for bias. By default, it is
            initialized with zeros if `with_bias` is `True`.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
    """
    def __init__(self, parents, name='', *args, **kwargs):
        mo.Linear.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name)
        if len(self._parents) > 1:
            raise RuntimeError


class DwConv(mo.DwConv, Module):
    r"""
    The DwConv module performs a depthwise convolution on the
    output of its parent. It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
        in_channels (:obj:`int`): Number of convolution kernels (which is
            equal to the number of input channels).
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For
            example, to apply convolution on an image with a 3 (height) by 5
            (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`, optional): Padding sizes for
            dimensions. Defaults to None.
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to None.
        dilation (:obj:`tuple` of :obj:`int`, optional): Dilation sizes for
            dimensions. Defaults to None.
        multiplier (:obj:`int`, optional): Number of output feature maps per
            input feature map. Defaults to 1.
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

    References:
        - F. Chollet: Chollet, Francois. "Xception: Deep Learning with
        Depthwise Separable Convolutions. https://arxiv.org/abs/1610.02357
    """
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.DwConv.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class MaxPool(mo.MaxPool, Module):
    r"""
    The MaxPool module performs max pooling on the
    output of its parent. It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
        kernel(:obj:`tuple` of :obj:`int`): Kernel sizes for each spatial axis.
        stride(:obj:`tuple` of :obj:`int`, optional): Subsampling factors for
            each spatial axis. Defaults to `None`.
        pad(:obj:`tuple` of :obj:`int`, optional): Border padding values for
            each spatial axis. Padding will be added both sides of the
            dimension. Defaults to ``(0,) * len(kernel)``.
        channel_last(bool): If True, the last dimension is considered as
            channel dimension, a.k.a NHWC order. Defaults to ``False``.
    """
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.MaxPool.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class AvgPool(mo.AvgPool, Module):
    r"""
    The AvgPool module performs avg pooling on the
    output of its parent. It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
        kernel(:obj:`tuple` of :obj:`int`): Kernel sizes for each spatial axis.
        stride(:obj:`tuple` of :obj:`int`, optional): Subsampling factors for
            each spatial axis. Defaults to `None`.
        pad(:obj:`tuple` of :obj:`int`, optional): Border padding values for
            each spatial axis. Padding will be added both sides of the
            dimension. Defaults to ``(0,) * len(kernel)``.
        channel_last(bool): If True, the last dimension is considered as
            channel dimension, a.k.a NHWC order. Defaults to ``False``.
    """
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.AvgPool.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class GlobalAvgPool(mo.GlobalAvgPool, Module):
    r"""
    The GlobalAvgPool module performs global avg pooling on the
    output of its parent. It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
    """
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.GlobalAvgPool.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class ReLU(mo.ReLU, Module):
    r"""
    The ReLu module is the static version of nnabla_nas.modules.ReLU.
    It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
        inplace (bool, optional): can optionally do the operation in-place.
            Default: ``False``.
    """
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.ReLU.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class Dropout(mo.Dropout, Module):
    r"""
    The Dropout module is the static version of nnabla_nas.modules.Dropout.
    It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
        drop_prob (:obj:`int`, optional): The probability of an element to be
            zeroed. Defaults to 0.5.
    """
    def __init__(self, parents, name='', *args, **kwargs):
        mo.Dropout.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name)
        if len(self._parents) > 1:
            raise RuntimeError


class BatchNormalization(mo.BatchNormalization, Module):
    r"""
    The BatchNormalization module is the static version of
    nnabla_nas.modules.BatchNormalization.
    It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
        n_features (int): Number of dimentional features.
        n_dims (int): Number of dimensions.
        axes (:obj:`tuple` of :obj:`int`):
            Mean and variance for each element in ``axes`` are calculated
            using elements on the rest axes. For example, if an input is 4
            dimensions, and ``axes`` is ``[1]``,  batch mean is calculated
            as ``np.mean(inp.d, axis=(0, 2, 3), keepdims=True)``
            (using numpy expression as an example).
        decay_rate (float, optional): Decay rate of running mean and
            variance. Defaults to 0.9
        eps (float, optional): Tiny value to avoid zero division by std.
            Defaults to 1e-5.
        output_stat (bool, optional): Output batch mean and variance.
            Defaults to `False`.
        fix_parameters (bool): When set to `True`, the beta and gamma will
            not be updated.
        param_init (dict):
            Parameter initializers can be set with a dict. A key of the
            dict must be ``'beta'``, ``'gamma'``, ``'mean'`` or ``'var'``.
            A value of the dict must be an :obj:`~nnabla.initializer.
            Initializer` or a :obj:`numpy.ndarray`.
            E.g. ``{
                    'beta': ConstantIntializer(0),
                    'gamma': np.ones(gamma_shape) * 2
                    }``.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    References:
        - Ioffe and Szegedy, Batch Normalization: Accelerating Deep
        Network Training by Reducing Internal Covariate Shift.
        https://arxiv.org/abs/1502.03167
    """
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.BatchNormalization.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class Merging(mo.Merging, Module):
    r"""
    The Merging module is the static version of
    nnabla_nas.modules.Merging.
    It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
        mode (str): The merging mode ('concat', 'add').
        axis (int, optional): The axis for merging when 'concat' is used.
            Defaults to 1.
    """
    def __init__(self, parents, mode, name='', eval_prob=None, axis=1):
        mo.Merging.__init__(self, mode, axis)
        Module.__init__(self, parents=parents, name=name,
                        eval_prob=eval_prob)
        if len(self._parents) == 1:
            raise RuntimeError


class Collapse(Module):
    r"""
    The Collapse module removes the last two
    singleton dimensions of an 4D input.
    It accepts only a single parent.

    Args:
        parents (list): the parents of this module
        name (string): the name of this module
    """
    def __init__(self, parents, name=''):
        Module.__init__(self, parents, name=name)
        if len(self._parents) > 1:
            raise RuntimeError

    def call(self, *inputs):
        return F.reshape(inputs[0],
                         shape=(inputs[0].shape[0],
                         inputs[0].shape[1]))


class Join(Module):
    r"""
    The Join module is used to fuse the output of multiple
    parents. It can either superpose them linearly, sample
    one of the input or select the maximum probable input.
    It accepts multiple parents. However,
    the output of all parents must have the same shape.

    Args:
        join_parameters (nnabla variable): a vector containing
            unnormalized categorical probabilities. It must have
            the same number of elements as the module has parents.
            The selection probability of each parent is calculated,
            using the softmax function.
        mode (string): can be 'linear'/'sample'/'max'. Determines
            how Join combines the output of the parents.
    """
    def __init__(self, parents, join_parameters, name='',
                 mode='linear', *args, **kwargs):
        if len(parents) < 2:
            raise Exception("Join vertice {} must have at "
                            "least 2 inputs, but has {}.".format(
                                    self.name, len(parents)))

        self._supported_modes = ('linear', 'sample', 'max')
        self.mode = mode

        if join_parameters.size == len(parents):
            self._join_parameters = join_parameters
        else:
            raise Exception(
                "The number of provided join parameters does not"
                " match the number of parents")
        self._sel_p = F.softmax(self._join_parameters)
        Module.__init__(self, parents=parents,
                        name=name, *args, **kwargs)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, m):
        if m in self._supported_modes:
            self._mode = m
        else:
            raise Exception("Join only supports the modes: {}".format(
                self._supported_modes))

    @property
    def _alpha(self):
        return self._join_parameters

    @_alpha.setter
    def _alpha(self, value):
        self._alpha = value

    def call(self, *input):
        """
        Aggregates all input tensors to one single
        input tensor (summing them up)
        """
        res = 0.0
        if self.mode == 'linear':
            for pi, inpi in zip(self._sel_p, *input):
                res += pi.reshape((1,)*len(inpi.shape)) * inpi
        elif self.mode == 'sample' or self.mode == 'max':
            res = input[0]
        return res

    def _recursive_call(self):
        if self._value is None:
            self.need_grad = True
            if self.mode == 'linear':
                self._value = self.call([pi() for pi in self.parents])
            elif self.mode == 'sample':
                self._sel_p.forward()
                self._idx = np.random.choice(
                    len(self.parents), 1, p=self._sel_p.d)[0]
                self._value = self.call(self.parents[self._idx]())

                # update the score function
                score = self._sel_p.d
                score[self._idx] -= 1
                self._join_parameters.g = score
                # print('{}/{}'.format(self.name,score[0]))
            elif self.mode == 'max':
                self._idx = np.argmax(self._join_parameters.d)
                self._value = self.call(self.parents[self._idx]())
        return self._value

    def _shape_function(self):
        if self.mode == 'linear':
            inputs = [nn.Variable(pi.shape) for pi in self.parents]
        elif self.mode == 'sample' or self.mode == 'max':
            inputs = nn.Variable(self.parents[0].shape)
        dummy_graph = self.call(inputs)
        return dummy_graph.shape


class Graph(mo.ModuleList, Module):
    r"""
    The static version of nnabla_nas.module.ModuleList.
    A Graph which can contain many modules. A graph can
    also be used as a module within another graph. Any graph
    must define self._output, i.e. the StaticModule which acts
    as the output node of this graph.
    """
    def __init__(self, parents=[],
                 name='', eval_prob=None,
                 *args, **kwargs):
        mo.ModuleList.__init__(self, *args, **kwargs)
        Module.__init__(self, name=name, parents=parents,
                        eval_prob=eval_prob)
        self._output = None

    @property
    def output(self):
        return self[-1]

    def _recursive_call(self):
        return self.output()

    @property
    def shape(self):
        """
        The output determines the shape of the graph.
        """
        return self.output.shape

    @shape.setter
    def shape(self, value):
        self.output.shape = value

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Graph(name=self._name + '/'+str(index),
                         parents=self._parents,
                         modules=list(self.modules.values())[index])
        index = _get_abs_string_index(self, index)
        return self.modules[index]

    def __delitem__(self, index):
        raise RuntimeError

    def reset_value(self):
        for mi in self.modules:
            try:
                self.modules[mi].reset_value()
            except Exception:
                pass

    def get_gv_graph(self, active_only=True,
                     color_map={Join: 'blue',
                                Merging: 'green',
                                Zero: 'red'}):
        r"""
        Construct a graphviz graph object that can be used
        to visualize the graph.

        Args:
            active_only (bool): whether or not to add inactive
                modules, i.e., modules which are not part of
                the computational graph
            color_map (dict): the mapping of class instance to
                vertice color used to visualize the graph.
        """
        graph = Digraph(name=self.name)
        # 1. get all the static modules in the graph
        if active_only:
            modules = [mi for _, mi in self.get_modules() if
                       isinstance(mi, Module) and
                       type(mi) != Graph and
                       mi._value is not None]
        else:
            modules = [mi for _, mi in self.get_modules()
                       if isinstance(mi, Module) and type(mi) != Graph]

        # 2. add these static modules as vertices to the graph
        for mi in modules:
            try:
                mi._eval_prob.forward()
            except Exception:
                pass
            caption = mi.name + "\n p: {:3.4f}ms".format(mi.eval_prob.d)
            try:
                graph.attr('node', color=color_map[type(mi)])
            except Exception:
                pass
            graph.node(mi.name, caption)

        # 3. add the edges
        for mi in modules:
            parents = mi.parents
            if len(parents) > 0:
                for pi in parents:
                    if active_only:
                        if pi.output._value is not None:
                            graph.edge(pi.output.name, mi.name,
                                       label=str(pi.output.shape))
                    else:
                        graph.edge(pi.output.name, mi.name,
                                   label=str(pi.output.shape))
        return graph
