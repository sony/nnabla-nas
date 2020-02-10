"""
Classes for graph based definition of DNN architectures and candidate spaces.
"""

import operator

import nnabla as nn
import nnabla.functions as F

#from nnabla_nas.graph.profiling import *
import numpy as np
from nnabla.initializer import ConstantInitializer
from nnabla.initializer import UniformInitializer
from nnabla.initializer import calc_uniform_lim_glorot

import nnabla_nas.contrib.misc as misc
import nnabla_nas.module as mo
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
        self._parent = None
        self._init_parent(parent)
        self._children      = []
        self._name          = name
        self._value         = None
        self._eval_probs    = None
        self._prof          = None
        self._shape         = None

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
            pass

    def _shape_function(self):
        #we build a nummy nnabla graph to infer the shapes
        input = nn.Variable(self.parent.shape)
        dummy_graph  = self.call(input)
        return dummy_graph.shape

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self._shape_function()
        return self._shape

    @property
    def input_shape(self):
       if hasattr(self._parent , '__iter__'):
          return [pi.shape for pi in self._parent]
       else:
          return [self._parent.shape]

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


    @property
    def training(self):
        r"""The training mode of module."""
        if '_training' not in self.__dict__:
            self.__dict__['_training'] = True
        return self._training

    @training.setter
    def training(self, mode):
        setattr(self, '_training', mode)
        for k, module in self.modules.items():
            if 'parent' not in k and 'child' not in k:
                module.training = mode

    def call(self, input): #
        raise NotImplementedError

    def clear_value(self):
        self._value = None

    def _recursive_call(self):
        if self._value is None:
            self._value = self.call(self.parent())
        #else:
        #    print("reusing graph for {}".format(self.name))
        return self._value

    def __call__(self):
        return self._recursive_call()

    @property
    def eval_prob(self):
        return self._eval_prob

    @eval_prob.setter
    def eval_prob(self, value):
        self._eval_prob = value

    def profile(self, estimator):
        return stop

    def reset_value(self):
        self._value = None

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
        if parent is not None:
            parent_type_mismatch = [not isinstance(pi, Module) for pi in parent]
            self._parent=[]
            if sum(parent_type_mismatch) == 0:
                for pi in parent:
                    self._parent.append(pi)
                    pi._children.append(self)
            else:
                raise Exception('At least one provided parent is not instance of class StaticModule')
        else:
            pass

    def _recursive_call(self): #
        return self.output()

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

    def reset_value(self):
        for mi in self.modules:
            try:
                self._value = None
                self.modules[mi].reset_value()
            except:
                pass #dynamic modules have no reset_value

#------------------------------------Some basic StaticModules------------------------------
class Input(Module):
    def __init__(self, name, value=None, eval_prob=None, *args, **kwargs):
        """
        An input op to the graph, which can store input values to propagate through the graph. If the input node has
        parent, it is the identity op, which just feeds the aggregated inputs to the output.
        """
        Module.__init__(self, name=name, parent=None, eval_prob=eval_prob)
        self._inp_value = value
        self._shape     = self._inp_value.shape

    def _init_parent(self, parent):
        """
        An input vertice can have no parents
        """
        self._parent = None

    def call(self, inputs):#
        return self._inp_value

    def _recursive_call(self): #
        return self.call(None)



class Identity(mo.Identity, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        Module.__init__(self, name, parent, eval_prob=eval_prob)

class Zero(mo.Zero, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.Zero.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)
        self._inp_value = nn.Variable.from_numpy_array(np.zeros(self._parent.shape))

    def call(self, input):
        return self._inp_value

    def _recursive_call(self):
        return self.call(None)


class Conv(mo.Conv, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.Conv.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

class Linear(mo.Linear, Module):
    def __init__(self, name, parent, *args, **kwargs):
        mo.Linear.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

class DwConv(mo.DwConv, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.DwConv.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

class SepConv(misc.SepConv, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        misc.SepConv.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

class MaxPool(mo.MaxPool, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.MaxPool.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

class AvgPool(mo.AvgPool, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.AvgPool.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

class GlobalAvgPool(mo.GlobalAvgPool, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.GlobalAvgPool.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

class ReLU(mo.ReLU, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.ReLU.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

class Dropout(mo.Dropout, Module):
    def __init__(self, name, parent, *args, **kwargs):
        mo.Dropout.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent)

class BatchNormalization(mo.BatchNormalization, Module):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        mo.BatchNormalization.__init__(self, *args, **kwargs)
        Module.__init__(self, name, parent, eval_prob=eval_prob)

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

    def _recursive_call(self):
        if self._value is None:
            self._value         = self.call(*[pi() for pi in self.parent])
        return self._value

    def _shape_function(self):
        #we build a nummy nnabla graph to infer the shapes
        inputs = [nn.Variable(pi.shape) for pi in self.parent]
        dummy_graph  = self.call(*inputs)
        return dummy_graph.shape

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
            self._join_parameters = join_parameters
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

    def call(self, input):
        """
        Aggregates all input tensors to one single input tensor (summing them up)
        """
        res = 0.0
        if self.mode == 'linear':
            for pi, inpi in zip(self._sel_p, input):
                res += pi.reshape((1,)*len(inpi.shape)) * inpi
        elif self.mode == 'sample' or self.mode == 'max':
            res = input
        return res

    def _recursive_call(self):
        if self._value is None:
            if self.mode == 'linear':
                self._value = self.call([pi() for pi in self.parent])
            elif self.mode == 'sample':
                self._sel_p.forward()
                self._idx = np.random.choice(len(self.parent), 1, p=self._sel_p.d)[0]
                self._value = self.call(self.parent[self._idx]())
            elif self.mode == 'max':
                self._idx = np.argmax(self._join_parameters.d)
                self._value = self.call(self.parent[self._idx]())
        #print(self._value)
        return self._value

    def _shape_function(self):
        #we build a nummy nnabla graph to infer the shapes
        inputs = [nn.Variable(pi.shape) for pi in self.parent]
        dummy_graph  = self.call(inputs)
        return dummy_graph.shape


if __name__ == '__main__':
    from nnabla_nas.module.static import NNablaProfiler
    from nnabla_nas.utils import LatencyEstimator

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

    prof = LatencyEstimator()
    lat = prof.predict(myGraph.modules['1'])
    print("latency of module {} is {}".format(myGraph.modules['1'].name, lat))

    eval_p = {gm: myGraph.modules[gm].eval_prob for gm in myGraph.modules}
    for evi in eval_p:
        try:
            eval_p[evi].forward()
        except:
            pass
        print("Module {} has evaluation probability {}".format(evi, eval_p[evi].d))

    import pdb; pdb.set_trace()
