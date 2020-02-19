"""
Defintion of static modules. Static modules are handy for Neural Architecture
Search (NAS), because they encode the graph structure.
"""

import operator

import nnabla as nn
import nnabla.functions as F
from nnabla.logger import logger
import numpy as np

import nnabla_nas.module as mo
from nnabla_nas.module.parameter import Parameter

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
    A static module encodes the graph structure, i.e., it has parents
    and children. Therefore, we can build graphs which can run simple
    graph optimizations when building the computational nnabla graph.
    """

    def __init__(self, parents=[], name='', eval_prob=None, *args, **kwargs):
        r"""
        :param parents: list, a list of static modules which are parents
        :param name: string, (optional) the name of the static module
        :param eval_prob: nnabla variable, (optional) the probability to
        evaluate this static module
        """
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
        adds a static_module as a child to self
        :param child: static_module, the module to add as a child
        """
        self._children.append(child)

    def _shape_function(self):
        r"""
        calculates the output shape of this static_module
        """
        inputs = [nn.Variable(pi.shape) for pi in self.parents]
        dummy_graph = self.call(*inputs)
        return dummy_graph.shape

    @property
    def shape(self):
        r"""
        the output shape of the static_module
        """
        if self._shape == -1:
            self._shape = self._shape_function()
        return self._shape

    @property
    def input_shapes(self):
        r"""
        returns a list of input shapes of this module
        """
        return [pi.shape for pi in self._parent]

    @property
    def name(self):
        r"""
        returns the name of the module
        """
        return self._name

    @property
    def parents(self):
        r"""
        returns a list of parent modules
        """
        return self._parents

    @property
    def children(self):
        r"""
        returns the child modules
        """
        return self._children

    def call(self, *inputs):
        r"""
        defines the transfer function of the module. Builds
        the computational graph of this module.
        :param input: one or multiple input variables
        """
        raise NotImplementedError

    def _recursive_call(self):
        r"""
        builds the computational graph by recursively calling
        the call methods of the parent modules.
        """
        if self._value is None:
            self._value = self.call(*[pi() for pi in self.parents])
            self.need_grad = True
        return self._value

    def __call__(self):
        r"""
        build the computational graph of to this module.
        """
        return self._recursive_call()

    @property
    def eval_prob(self):
        r"""
        The evaluation probability of this module. It is
        1.0 if not specified otherwise.
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
        """
        return self

    def reset_value(self):
        r"""
        Resets all _values, need_grad flags and shapes
        """
        self._value = None
        self.apply(need_grad=False)
        self._shape = -1


class Input(Module):
    r"""
    A static module that can serve as an input, i.e., it has no parents
    but is provided with a value which it can pass to its children
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
        return self._value

    def _recursive_call(self):
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
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        Module.__init__(self, parents, name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError

class Zero(mo.Zero, Module):
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
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.Conv.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name,  eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class Linear(mo.Linear, Module):
    def __init__(self, parents, name='', *args, **kwargs):
        mo.Linear.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name)
        if len(self._parents) > 1:
            raise RuntimeError


class DwConv(mo.DwConv, Module):
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.DwConv.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class MaxPool(mo.MaxPool, Module):
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.MaxPool.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class AvgPool(mo.AvgPool, Module):
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.AvgPool.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class GlobalAvgPool(mo.GlobalAvgPool, Module):
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.GlobalAvgPool.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class ReLU(mo.ReLU, Module):
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.ReLU.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class Dropout(mo.Dropout, Module):
    def __init__(self, parents, name='', *args, **kwargs):
        mo.Dropout.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name)
        if len(self._parents) > 1:
            raise RuntimeError


class BatchNormalization(mo.BatchNormalization, Module):
    def __init__(self, parents, name='', eval_prob=None, *args, **kwargs):
        mo.BatchNormalization.__init__(self, *args, **kwargs)
        Module.__init__(self, parents, name=name, eval_prob=eval_prob)
        if len(self._parents) > 1:
            raise RuntimeError


class Merging(mo.Merging, Module):
    def __init__(self, parents, mode, name='', eval_prob=None, axis=1):
        mo.Merging.__init__(self, mode, axis)
        Module.__init__(self, parents=parents, name=name, 
                        eval_prob=eval_prob)
        if len(self._parents) == 1:
            raise RuntimeError


class Collapse(Module):
    r"""
    Collapses the last two singleton dimensions of an 4D input.
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
            if len(parents)>0:
                for pi in parents:
                    if active_only:
                        if pi.output._value is not None:
                            graph.edge(pi.output.name, mi.name,
                                       label=str(pi.output.shape))
                    else:
                        graph.edge(pi.output.name, mi.name,
                                   label=str(pi.output.shape))
        return graph


if __name__ == '__main__':

    input = Input(value=nn.Variable((10, 20, 32, 32)))
    
    import pdb; pdb.set_trace()

    class MyGraph(Graph):
        def __init__(self, name, parents):
            Graph.__init__(self, name=name, parents=parents)

            join_param = Parameter(shape=(3,))
            join_prob = F.softmax(join_param)

            self.append(Input(name='input_2',
                              value=nn.Variable((10, 20, 32, 32)),
                              eval_prob=join_prob[0]+join_prob[1]))
            self.append(Conv(name='conv', parent=self[-1], in_channels=20,
                             out_channels=20, kernel=(3, 3), pad=(1, 1),
                             eval_prob=join_prob[0]))
            self.append(Input(name='input_3',
                              value=nn.Variable((10, 20, 32, 32)),
                              eval_prob=join_prob[2]))
            self.append(Join(name='join',
                             parents=self,
                             mode='linear',
                             join_parameters=join_param))

    input_module_1 = Input(name='input_1', value=nn.Variable((10, 20, 32, 32)))
    myGraph = MyGraph(name='myGraph', parents=[input_module_1])
    out = myGraph()

    sliced = myGraph[:1]

    prof = LatencyEstimator()
    lat = prof.predict(myGraph.modules['1'])
    print("latency of module {} is {}".format(myGraph.modules['1'].name, lat))

    eval_p = {gm: myGraph.modules[gm].eval_prob for gm in myGraph.modules}
    for evi in eval_p:
        try:
            eval_p[evi].forward()
        except Exception as ex:
            logger.warning(str(ex))
        print("Module {} has evaluation probability {}".format(
            evi, eval_p[evi].d))

    import pdb
    pdb.set_trace()
