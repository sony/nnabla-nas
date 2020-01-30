"""
Classes for graph based definition of DNN architectures and candidate spaces.
"""

#from nnabla_nas.graph.profiling import *
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.initializer import (ConstantInitializer, UniformInitializer,
                                calc_uniform_lim_glorot)
from nnabla_nas.module.parameter import Parameter
import nnabla_nas.module as mo
import nnabla_nas.contrib.misc as misc
import operator

def _get_abs_string_index(obj, idx):
    """Get the absolute index for the list of modules"""
    idx = operator.index(idx)
    if not (-len(obj) <= idx < len(obj)):
        raise IndexError('index {} is out of range'.format(idx))
    if idx < 0:
        idx += len(obj)
    return str(idx)

class Module(mo.Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
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
        self._prof          = None
        self._shape         = None
        self._forward_tag   = False
        if eval_prob is None:
            self._eval_prob = nn.Variable.from_numpy_array(np.array(1.0))
        else:
            self._eval_prob = eval_prob

        mo.Module.__init__(self)

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

    def call(self, tag=None):
        if tag is None or self._forward_tag != tag:
            self._forward_tag   = not(self._forward_tag) #flip the tag
            self._value         = self._value_function(self.parent(self._forward_tag))
        return self._value

    def __call__(self, *args, **kargs):
        return self.call(*args, **kargs)

    @property
    def eval_prob(self):
        return self._eval_prob

    @eval_prob.setter
    def eval_prob(self, value):
        self._eval_prob = value

    #def _eval_prob_function(self, input):
    #    res = {}
    #
    #    for ii in input:
    #        res[ii] = input[ii]
    #
    #    res.update({self: nn.Variable.from_numpy_array(np.array(1.0))})
    #    return res

    def profile(self, profiler, n_run=100):
        if self._prof is None:
            self._prof = self._profile(profiler, n_run)
        return self._prof

    def _profile(self, profiler, n_run=100):
        input = nn.Variable(shape=self.parent.shape)
        out = self._value_function(input)
        try:
            return float(profiler.profile(out, n_run=n_run))
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

    def get_exp_latency(self, profiler, n_run=10):
        return self.profile(profiler, n_run=n_run)*self.eval_prob

#------------------------------------A graph of StaticModules--------------------------------------
class Graph(mo.ModuleList, Module):
    # Graph is derived from Op, such that we can realize nested graphs!
    def __init__(self, name, parents, eval_prob=None, *args, **kwargs):
        mo.ModuleList.__init__(self, *args, **kwargs)
        Module.__init__(self, name=name, parent=parents, eval_prob=eval_prob)
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

    def call(self, tag=None):
        return self.output(tag=tag)

    #def eval_probs(self, clear_probs=False):
    #    return self.output.eval_probs(clear_probs=clear_probs)

    @property
    def shape(self):
        """
        The output determines the shape of the graph.
        """
        if self._shape is None:
            self._shape = self.output.shape
        return self._shape

    def _profile(self, profiler, n_run=100):
        """
        Compared to a single vertice, we need to profile all vertices which are nested in this graph.
        """
        result = {}
        for mi in self.get_modules():
            if isinstance(mi[1], Module):
                if mi[1] != self and not isinstance(mi[1], Graph):
                    result[mi[1].name] = mi[1].profile(profiler, n_run=n_run)
                else:
                    print("do not profile {}".format(mi[1].name))
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

    def get_exp_latency(self, profiler, n_run=10):
        exp_latency = 0
        for mi in self.modules:
            exp_latency += self.modules[mi].get_exp_latency(profiler, n_run)
        return exp_latency

#------------------------------------Some basic StaticModules------------------------------
class Input(Module):
    def __init__(self, name, value=None, eval_prob=None, *args, **kwargs):
        """
        An input op to the graph, which can store input values to propagate through the graph. If the input node has
        parent, it is the identity op, which just feeds the aggregated inputs to the output.
        """
        Module.__init__(self, name=name, parent=None, eval_prob=eval_prob)
        self._value     = value
        self._shape     = self._value.shape

    def _init_parent(self, parent):
        """
        An input vertice can have no parents
        """
        self._parent = None

    def _value_function(self, inputs):
        return self._value

    #def eval_probs(self, clear_probs=False):
    #    if clear_probs:
    #        self.clear_eval_probs()
#
#        if self._eval_probs is None:
#            self._eval_probs = self._eval_prob_function(input={})
#        return self._eval_probs

#    def _eval_prob_function(self, input):
#        #The input module has no parents, therefore we only return the evaluation probability of self
#        return {self: nn.Variable.from_numpy_array(np.array(1.0))}

    def clear_value(self):
        """
        We are not allowed to clear the value of an input vertice!
        """
        pass

    def _profile(self, profiler, n_run=100):
        return 0.0

    def call(self, tag=None):
        return self._value_function(None)

class Identity(mo.Identity, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.Identity.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

    def _value_function(self, input):
        return mo.Identity.call(self, input)

    def call(self, tag=None):
        return Module.call(self, tag=tag)

class Zero(mo.Zero, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.Zero.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)
        self._value = nn.Variable.from_numpy_array(np.zeros(self._parent.shape))

    def _value_function(self, input):
        return self._value

    def call(self, tag=None):
        self._forward_tag = tag
        return self._value_function(None)

    #def _eval_prob_function(self, input): #a zero operation sets the evaluation pobabilities of all parents to 0
    #    return {self: nn.Variable.from_numpy_array(np.array(1.0))}

class Conv(mo.Conv, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.Conv.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

    def _value_function(self, input):
        return mo.Conv.call(self, input) #change to self.call

    def call(self, tag=None):
        return Module.call(self, tag=tag)

class Linear(mo.Dropout, Module):
    def __init__(self, name, parent, *args, **kwargs):
        mo.Linear.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return mo.Linear.call(self, input)

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

class DwConv(mo.DwConv, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.DwConv.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

    def _value_function(self, input):
        return mo.DwConv.call(self, input)

    def call(self, tag=None):
        return Module.call(self, tag=tag)

class SepConv(misc.SepConv, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        misc.SepConv.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

    def _value_function(self, input):
        return misc.SepConv.call(self, input)

    def call(self, tag=None):
        return Module.call(self, tag=tag)

class MaxPool(mo.MaxPool, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.MaxPool.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

    def _value_function(self, input):
        return mo.MaxPool.call(self, input)

    def call(self, tag=None):
        return Module.call(self, tag=tag)

class AvgPool(mo.AvgPool, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.AvgPool.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

    def _value_function(self, input):
        return mo.AvgPool.call(self, input)

    def call(self, tag=None):
        return Module.call(self, tag=tag)

class GlobalAvgPool(mo.GlobalAvgPool, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.GlobalAvgPool.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

    def _value_function(self, input):
        return mo.GlobalAvgPool.call(self, input)

    def call(self, tag=None):
        return Module.call(self, tag=tag)

class ReLU(mo.ReLU, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.ReLU.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

    def _value_function(self, input):
        return mo.ReLU.call(self, input)

    def call(self, tag=None):
        return Module.call(self, tag=tag)

class Dropout(mo.Dropout, Module):
    def __init__(self, name, parent, *args, **kwargs):
        mo.Dropout.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

    def _value_function(self, input):
        return mo.Dropout.call(self, input)

    def call(self, clear_value=False):
        return Module.call(self, clear_value=clear_value)

class BatchNormalization(mo.BatchNormalization, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.BatchNormalization.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

    def _value_function(self, input):
        return mo.BatchNormalization.call(self, input)

    def call(self, tag=None):
        return Module.call(self, tag=tag)

class Merging(mo.Merging, Module):
    def __init__(self, name, parents, mode, eval_prob=None, axis=1):
        mo.Merging.__init__(self, mode, axis)
        Module.__init__(self, name, parents, eval_prob=eval_prob)

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

    def call(self, tag=None):
        if tag is None or self._forward_tag != tag:
            self._forward_tag   = not(self._forward_tag) #flip the tag
            self._value         = self._value_function([pi(self._forward_tag) for pi in self.parent])
        return self._value

    #def eval_probs(self, clear_probs=False):
    #    if clear_probs:
    #        self.clear_eval_probs()
#
#        if self._eval_probs is None:
#            self._eval_probs = self._eval_prob_function([pi.eval_probs(clear_probs) for pi in self.parent])
#        return self._eval_probs

 #   def _eval_prob_function(self, input):
 #       res = {}
 #       for i, inpi in enumerate(input):  # dictionary
 #           for inpii in inpi: #dictionary element
 #               if inpii in res:
 #                   res[inpii].append(inpi[inpii])
 #               else:
 #                   res.update({inpii: [inpi[inpii]]})
#
#        for resi in res:
#            res[resi] = 1.0 - F.prod(1.0 - F.stack(*res[resi]))
#
#        res.update({self: nn.Variable.from_numpy_array(np.array(1.0))})
#        return res

    def _value_function(self, input):
        return mo.Merging.call(self,*input)

    def _shape_function(self):
        #we build a nummy nnabla graph to infer the shapes
        inputs = [nn.Variable(pi.shape) for pi in self.parent]
        dummy_graph  = self._value_function(inputs)
        return dummy_graph.shape

    def _profile(self, profiler, n_run=100):
        inputs = [nn.Variable(shape=pi.shape) for pi in self.parent]
        out = self._value_function(inputs)
        try:
            return float(profiler.profile(out, n_run=n_run))
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
            self._join_parameters = F.reshape(join_parameters, shape=(len(parents),), inplace=False)
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

    def call(self, tag=None):
        if tag is None or self._forward_tag != tag:
            self._forward_tag   = not(self._forward_tag) #flip the tag
            self._value         = self._value_function([pi(self._forward_tag) for pi in self.parent])
        return self._value

    #def eval_probs(self, clear_probs=False):
    #    if clear_probs:
    #        self.clear_eval_probs()
#
#        if self._eval_probs is None:
#            self._eval_probs = self._eval_prob_function([pi.eval_probs(clear_probs) for pi in self.parent])
#        return self._eval_probs

 #   def _eval_prob_function(self, input):
 #       res = {}
 #       for i, inpi in enumerate(input):  # dictionary
 #           for inpii in inpi: #dictionary element
 #               if inpii in res:
 #               # we need to multiply all the evaluation probabilities of one input with the corresponding selection #probability
 #                   res[inpii] += inpi[inpii] * self._sel_p[i]
 #               else:
 #                   res.update({inpii: inpi[inpii] * self._sel_p[i]})
 #       res.update({self: nn.Variable.from_numpy_array(np.array(1.0))})
 #       return res

    def _shape_function(self):
        #we build a nummy nnabla graph to infer the shapes
        inputs = [nn.Variable(pi.shape) for pi in self.parent]
        dummy_graph  = self._value_function(inputs)
        return dummy_graph.shape

    def _profile(self, profiler, n_run=100):
        inputs = [nn.Variable(shape=pi.shape) for pi in self.parent]
        out = self._value_function(inputs)
        try:
            return float(profiler.profile(out, n_run=n_run))
        except:
            print("Cannot profile module {}!".format(self.name))
            return 0.0


if __name__ == '__main__':
    from nnabla_nas.module.static import NNablaProfiler
    class MyGraph(Graph):
        def __init__(self, name, parents):
            Graph.__init__(self, name=name, parents=parents)

            join_param = Parameter(shape=(3,))
            join_prob  = F.softmax(join_param)

            self.append(Input(name='input_2', value=nn.Variable((10,20,32,32)),
                        eval_prob=join_prob[0]+join_prob[1]))
            self.append(Conv(name='conv', parent=self[-1], in_channels=20, out_channels=20, kernel=(3,3), pad=(1,1),
                        eval_prob=join_prob[0]))
            self.append(Input(name='input_3', value=nn.Variable((10,20,32,32)),
                        eval_prob=join_prob[2]))
            self.append(Join(name='join',
                             parents=self,
                             mode='linear',
                             join_parameters=join_param))
            #self.append(Join(name='join', parent=self[-1], join_parameters=Parameter(shape=(2,))))
#            self.append(Merge(name='merge', parent=[self[-1], self[0]]))

    input_module_1 = Input(name='input_1', value=nn.Variable((10,20,32,32)))
    myGraph = MyGraph(name='myGraph', parents=[input_module_1])
    out = myGraph()

    sliced = myGraph[:1]

    latency = myGraph.profile(NNablaProfiler(), n_run=10)

    eval_p = {gm: myGraph.modules[gm].eval_prob for gm in myGraph.modules}
    for evi in eval_p:
        try:
            eval_p[evi].forward()
        except:
            pass
        print("Module {} has evaluation probability {}".format(evi, eval_p[evi].d))

    print(latency)
    import pdb; pdb.set_trace()

