"""
Classes for graph based definition of DNN architectures and candidate spaces.
"""

import operator

import nnabla as nn
import nnabla.functions as F
#from nnabla_nas.graph.profiling import *
import numpy as np
from nnabla.initializer import (ConstantInitializer, UniformInitializer,
                                calc_uniform_lim_glorot)

import nnabla_nas.contrib.misc as misc
import nnabla_nas.module as smo
from nnabla_nas.module.parameter import Parameter


def _get_abs_string_index(obj, idx):
    """Get the absolute index for the list of modules"""
    idx = operator.index(idx)
    if not (-len(obj) <= idx < len(obj)):
        raise IndexError('index {} is out of range'.format(idx))
    if idx < 0:
        idx += len(obj)
    return str(idx)

class Module(smo.Module):
    def __init__(self, name, parent, *args, **kwargs):
        print('init {}'.format(name))
        """
        A vertice is a computational node in a DNN architecture.
        It has a unique name, and a reference to a graph object where it lives in.
        You can provide collections (a list of strings). A collection is a group
        of vertices which share common properties or belong together because of
        a certain reason.

        :param name: string, the unique name of the vertice which identifies it.
        The name must be unique within the graph the vertice lives in.
        :param parent: list, list of parent vertices
        :collections: list of strings, the collections this vertice belongs to.
        A collection is a group of vertices, which belong together for some
        reason.
        """
        self._init_parent(parent)
        self._children      = []
        self._name          = name
        self._value         = None
        self._eval_probs    = None
        self._shape         = None

        smo.Module.__init__(self)

    def add_child(self, child):
        self._children.append(child)

    def _init_parent(self, parent):
        if isinstance(parent, Module):
            self._parent = parent
            self._parent.add_child(self)
        else:
            raise RuntimeError

    def _shape_function(self):
        #we build a nummy nnabla graph to infer the shapes
        input = nn.Variable(self.parent.shape)
        dummy_graph  = self._value_function(input)
        return dummy_graph.shape

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self._shape_function()
        return self._shape

    @property
    def name(self):
        """
        A unique name within the graph this vertice lives in
        """
        return self._name

    @property
    def parent(self):
        """
        The parent vertices of this vertice (all vertices which are connected
        with incoming edges).
        """
        #import pdb; pdb.set_trace()
        return self._parent

    @property
    def children(self):
        """
        The vhild vertices of this vertice (all vertices which are connected
        with outgoing edges).
        """
        return self._children

    def _value_function(self, input):
        raise NotImplementedError

    def clear_value(self):
        self._value = None

    def call(self, clear_value=False):
        print("calling "+self.name)
        if clear_value:
            self.clear_value()

        if self._value is None:
            self._value = self._value_function(self.parent(clear_value))
        return self._value

    def clear_eval_probs(self):
        self._eval_probs = None

    def eval_probs(self, clear_probs=False):
        if clear_probs:
            self.clear_eval_probs()

        if self._eval_probs is None:
            self._eval_probs = self._eval_prob_function(self.parent.eval_probs(clear_probs))
        return self._eval_probs

    def _eval_prob_function(self, input):
        input.update({self: nn.Variable.from_numpy_array(np.array(1.0))})
        return input

    def profile(self, profiler, n_run=100):
        input = nn.Variable(shape=self.parent.shape)
        out = self._value_function(input)
        try:
            return profiler.profile(out, n_run=n_run)
        except:
            print("Cannot profile module {}!".format(self.name))
            return 0.0

    def param_memory_size(self, bitwidth=32, div=8*2**20):
        """
        Returns the memory requirement to store the parameters of the vertice
        :param bitwidth: integer, the bitwidth used to encode the parameters
        :param div: integer, e.g. 8*2**20 if we want to have the complexity in MB
        """
        mem = 0.0
        params = self.get_parameters()
        for pi in params:
            mem += bitwidth * np.prod(params[pi].shape) / div
        return mem

    def activation_memory_size(self, bitwidth=32, div=8*2**20):
        """
        Returns the memory requirement to store the activations of the vertice
        :param bitwidth: integer, the bitwidth used to encode the parameters
        :param div: integer, e.g. 8*2**20 if we want to have the complexity in MB
        """
        return bitwidth * np.prod(self.shape) / div

#------------------------------------A graph of StaticModules--------------------------------------
class Graph(smo.ModuleList, Module):
    # Graph is derived from Op, such that we can realize nested graphs!
    def __init__(self, name, parents, *args, **kwargs):
        smo.ModuleList.__init__(self, *args, **kwargs)
        Module.__init__(self, name=name, parent=parents)
        self._output = None

    @property
    def output(self):
        return self[-1]

    def _init_parent(self, parent):
        parent_type_mismatch = [not isinstance(pi, Module) for pi in parent]
        self._parent=[]
        if sum(parent_type_mismatch) == 0:
            for pi in parent:
                self._parent.append(pi)
                pi._children.append(self)
        else:
            raise Exception('At least one provided parent is not instance of class StaticModule')

    def clear_value(self):
        for gmi in self._graph_modules:
            gmi.clear_value()

    def clear_probs(self):
        for gmi in self._graph_modules:
            gmi.clear_value()

    def call(self, clear_value=False):
        return self.output(clear_value=clear_value)

    def eval_probs(self, clear_probs=False):
        return self.output.eval_probs(clear_probs=clear_probs)

    @property
    def shape(self):
        """
        The output determines the shape of the graph.
        """
        if self._shape is None:
            self._shape = self.output.shape
        return self._shape

    def profile(self, profiler, n_run=100):
        """
        Compared to a single vertice, we need to profile all vertices which are nested in this graph.
        """
        result = {}
        for mi in self.get_modules():
            if isinstance(mi[1], Module) and mi[1] != self:
                result[mi[1].name] = mi[1].profile(profiler, n_run=n_run)
        return result

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Graph(name=self._name+ '/'+str(index),
                                  parents=self._parent,
                                  modules=list(self.modules.values())[index])
        index = _get_abs_string_index(self, index)
        return self.modules[index]

    def __delitem__(self, index):
        raise RuntimeError

#------------------------------------Some basic StaticModules------------------------------
class Input(Module):
    def __init__(self, name, value=None, *args, **kwargs):
        """
        An input op to the graph, which can store input values to propagate through the graph. If the input node has
        parent, it is the identity op, which just feeds the aggregated inputs to the output.
        """
        Module.__init__(self, name=name, parent=None)
        self._value     = value
        self._shape     = self._value.shape

    def _init_parent(self, parent):
        """
        An input vertice can have no parents
        """
        self._parent = None

    def _value_function(self, inputs):
        return self._value

    def eval_probs(self, clear_probs=False):
        if clear_probs:
            self.clear_eval_probs()

        if self._eval_probs is None:
            self._eval_probs = self._eval_prob_function(input={})
        return self._eval_probs

    def _eval_prob_function(self, input):
        #The input module has no parents, therefore we only return the evaluation probability of self
        return {self: nn.Variable.from_numpy_array(np.array(1.0))}

    def clear_value(self):
        """
        We are not allowed to clear the value of an input vertice!
        """
        pass

    def profile(self, profiler, n_run=100):
        return 0.0

class Identity(smo.Identity, Module):
    def __init__(self, name, parent, *args, **kwargs):
        smo.Identity.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return smo.Identity.call(self, input)

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

class Zero(smo.Zero, Module):
    def __init__(self, name, parent, *args, **kwargs):
        smo.Zero.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return smo.Zero.call(self, input)

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

    def _eval_prob_function(self, input): #a zero operation sets the evaluation pobabilities of all parents to 0
        return {self: nn.Variable.from_numpy_array(np.array(1.0))}

class Conv(smo.Conv, Module):
    def __init__(self, name, parent, *args, **kwargs):
        smo.Conv.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return smo.Conv.call(self, input) #change to self.call

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

class DwConv(smo.DwConv, Module):
    def __init__(self, name, parent, *args, **kwargs):
        smo.DwConv.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return smo.DwConv.call(self, input)

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

class SepConv(misc.SepConv, Module):
    def __init__(self, name, parent, *args, **kwargs):
        misc.SepConv.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return misc.SepConv.call(self, input)

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

class MaxPool(smo.MaxPool, Module):
    def __init__(self, name, parent, *args, **kwargs):
        smo.MaxPool.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return smo.MaxPool.call(self, input)

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

class AvgPool(smo.AvgPool, Module):
    def __init__(self, name, parent, *args, **kwargs):
        smo.AvgPool.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return smo.AvgPool.call(self, input)

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

class GlobalAvgPool(smo.GlobalAvgPool, Module):
    def __init__(self, name, parent, *args, **kwargs):
        smo.GlobalAvgPool.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return smo.GlobalAvgPool.call(self, input)

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

class ReLU(smo.ReLU, Module):
    def __init__(self, name, parent, *args, **kwargs):
        smo.ReLU.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return smo.ReLU.call(self, input)

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

class BatchNormalization(smo.BatchNormalization, Module):
    def __init__(self, name, parent, *args, **kwargs):
        smo.BatchNormalization.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return smo.BatchNormalization.call(self, input)

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

class Merging(smo.Merging, Module):
    def __init__(self, name, parents, mode, axis=1):
        smo.Merging.__init__(self, mode, axis)
        Module.__init__(self, name, parents)

    def _init_parent(self, parent):
        #we allow for multiple parents!
        parent_type_mismatch = [not isinstance(pi, Module) for pi in parent]
        self._parent=[]
        if sum(parent_type_mismatch) == 0:
            for pi in parent:
                self._parent.append(pi)
                pi._children.append(self)
        else:
            raise Exception('At least one provided parent is not a static module!')

    def call(self, clear_value=False):
        print("calling "+self.name)
        if clear_value:
            self.clear_value()

        if self._value is None:
            self._value = self._value_function([pi(clear_value) for pi in self.parent])
        return self._value

    def eval_probs(self, clear_probs=False):
        if clear_probs:
            self.clear_eval_probs()

        if self._eval_probs is None:
            self._eval_probs = self._eval_prob_function([pi.eval_probs(clear_probs) for pi in self.parent])
        return self._eval_probs

    def _eval_prob_function(self, input):
        res = {}
        for i, inpi in enumerate(input):  # dictionary
            for inpii in inpi: #dictionary element
                if inpii in res:
                    res[inpii].append(inpi[inpii])
                else:
                    res.update({inpii: [inpi[inpii]]})

        for resi in res:
            res[resi] = 1.0 - F.prod(1.0 - F.stack(*res[resi]))

        res.update({self: nn.Variable.from_numpy_array(np.array(1.0))})
        return res

    def _value_function(self, input):
        return smo.Merging.call(self,*input)

    def _shape_function(self):
        #we build a nummy nnabla graph to infer the shapes
        inputs = [nn.Variable(pi.shape) for pi in self.parent]
        dummy_graph  = self._value_function(inputs)
        return dummy_graph.shape

    def profile(self, profiler, n_run=100):
        inputs = [nn.Variable(shape=pi.shape) for pi in self.parent]
        out = self._value_function(inputs)
        try:
            return profiler.profile(out, n_run=n_run)
        except:
            print("Cannot profile module {}!".format(self.name))
            return 0.0


class Join(Module):
    def __init__(self, name, parents, join_parameters, mode='linear', *args, **kwargs):
        """
        :param join_parameters: nnabla variable of shape (#parents,1), the logits for the join operations
        :param mode: string, the mode we use to join (linear, sample, max)
        """
        if len(parents) < 2:
            raise Exception("Join vertice {} must have at least 2 inputs, but has {}.".format(self.name, len(parent)))

        self._supported_modes =['linear', 'sample', 'max']
        self.mode = mode

        if join_parameters.size == len(parents):
            self._join_parameters = F.reshape(join_parameters, shape=(len(parents),))
        else:
            raise Exception("The number of provided join parameters does not match the number of parents")
        self._sel_p = F.softmax(self._join_parameters)

        Module.__init__(self, name=name, parent=parents, *args, **kwargs)

    def _init_parent(self, parent):
        #we allow for multiple parents!
        parent_type_mismatch = [not isinstance(pi, Module) for pi in parent]
        self._parent=[]
        if sum(parent_type_mismatch) == 0:
            for pi in parent:
                self._parent.append(pi)
                pi._children.append(self)
        else:
            raise Exception('At least one provided parent is not a static module!')

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, m):
        if m in self._supported_modes:
            self._mode = m
        else:
            raise Exception("Join only supports the modes: {}".format(self._supported_modes))

    def _value_function(self, input):
        """
        Aggregates all input tensors to one single input tensor (summing them up)
        """
        def one_hot(x, n=len(input)):
            return np.array([int(i == x) for i in range(n)])

        res = 0.0
        if self.mode == 'linear':
            for pi, inpi in zip(self._sel_p, input):
                res += pi.reshape((1,)*len(inpi.shape)) * inpi
        elif self.mode == 'sample':
            self._sel_p.forward()
            self._idx = np.random.choice(len(inputs), 1, p=self._sel_p.d)[0]
            print('{} selects input {} with p={}'.format(self.name, self._idx, self._sel_p.d[self._idx]))
            res = inputs[self._idx]
            self._z = one_hot(self._idx)
            self._score = self._z - self._sel_p.d
        elif self.mode == 'max':
            #just pick the input channel with the highest probability!
            self._idx = np.argmax(self._join_parameters.d)
            res = inputs[self._idx]
            self._z = one_hot(self._idx)
            self._score = self._z - self._sel_p.d
            print('{} selects input {}'.format(self.name, self._idx))
        return res

    def call(self, clear_value=False):
        print("calling "+self.name)
        if clear_value:
            self.clear_value()

        if self._value is None:
            self._value = self._value_function([pi(clear_value) for pi in self.parent])
        return self._value

    def eval_probs(self, clear_probs=False):
        if clear_probs:
            self.clear_eval_probs()

        if self._eval_probs is None:
            self._eval_probs = self._eval_prob_function([pi.eval_probs(clear_probs) for pi in self.parent])
        return self._eval_probs

    def _eval_prob_function(self, input):
        res = {}
        for i, inpi in enumerate(input):  # dictionary
            for inpii in inpi: #dictionary element
                if inpii in res:
                # we need to multiply all the evaluation probabilities of one input with the corresponding selection probability
                    res[inpii] += inpi[inpii] * self._sel_p[i]
                else:
                    res.update({inpii: inpi[inpii] * self._sel_p[i]})
        res.update({self: nn.Variable.from_numpy_array(np.array(1.0))})
        return res

    def _shape_function(self):
        #we build a nummy nnabla graph to infer the shapes
        inputs = [nn.Variable(pi.shape) for pi in self.parent]
        dummy_graph  = self._value_function(inputs)
        return dummy_graph.shape

    def profile(self, profiler, n_run=100):
        inputs = [nn.Variable(shape=pi.shape) for pi in self.parent]
        out = self._value_function(inputs)
        try:
            return profiler.profile(out, n_run=n_run)
        except:
            print("Cannot profile module {}!".format(self.name))
            return 0.0


if __name__ == '__main__':
    from nnabla_nas.module.static import NNablaProfiler
    class MyGraph(Graph):
        def __init__(self, name, parents):
            Graph.__init__(self, name=name, parents=parents)
            self.append(Input(name='input_2', value=nn.Variable((10,20,32,32))))
            self.append(SepConv(name='conv', parent=self[-1], in_channels=20, out_channels=20, kernel=(3,3), pad=(1,1)))
            self.append(Input(name='input_3', value=nn.Variable((10,20,32,32))))
            self.append(Join(name='concat',
                             parents=self,
                             mode='linear',
                             join_parameters=Parameter(shape=(3,))))
            #self.append(Join(name='join', parent=self[-1], join_parameters=Parameter(shape=(2,))))
#            self.append(Merge(name='merge', parent=[self[-1], self[0]]))

    input_module_1 = Input(name='input_1', value=nn.Variable((10,20,32,32)))
    myGraph = MyGraph(name='myGraph', parents=[input_module_1])
    out = myGraph()

    sliced = myGraph[:1]

    latency = myGraph.profile(NNablaProfiler(), n_run=10)

    eval_p = myGraph.eval_probs()
    for evi in eval_p:
        try:
            eval_p[evi].forward()
        except:
            pass
        print("Module {} has evaluation probability {}".format(evi.name, eval_p[evi].d))

    print(latency)
    import pdb; pdb.set_trace()
