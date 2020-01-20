"""
Classes for graph based definition of DNN architectures and candidate spaces.
"""

#from nnabla_nas.graph.profiling import *
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla_nas.module import Module
from nnabla_nas.module.module import ModuleDict
from nnabla_nas.module.convolution import Conv
from nnabla_nas.module.parameter import Parameter

class StaticModule(Module):
    def __init__(self, name, parent, *args, **kwargs):
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
        self._parent        = []
        self._init_parents(parent)

        self._child         = []
        self._name          = name
        self._profiler      = None
        self._value         = None
        self._eval_prob     = None
        self._shape         = None

        if not self._shapes_valid(self._parent):
            raise Exception("Input shapes of vertice {} are not valid!".format(self.name))

        self._create_modules()
        super(StaticModule, self).__init__()

    def _shapes_valid(self, parent):
        shapes      = [(list(pi.shape) + 4 * [1])[:4] for pi in parent]
        if len(shapes) > 1:
            if shapes.count(shapes[0]) == len(shapes):
                return True
            else:
                return False
        else:
            return True

    def _shape_function(self, parent):
        return (None,)*4

    def _init_parents(self, parent):
        if len(parent) > 0:
            parent_type_mismatch = [not isinstance(pi, StaticModule) for pi in parent]
            parent_shape_mismatch = []
            if sum(parent_type_mismatch) == 0:
                for pi in parent:
                    self._parent.append(pi)
                    pi._child.append(self)
            else:
                raise Exception('At least one provided parent is not instance of class Vertice')

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self._shape_function(self.parent)
        return self._shape

    def _create_modules(self):
	    pass

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
    def child(self):
        """
        The vhild vertices of this vertice (all vertices which are connected
        with outgoing edges).
        """
        return self._child

    def _value_function(self, input):
        raise NotImplementedError

    def _aggregate_inputs(self, inputs):
        """
        Aggregates all input tensors to one single input tensor (summing them up)
        """
        if len(inputs) > 1:
            try:
                res = 0
                for ii in inputs:
                    res += ii
            except:
                raise Exception("Inputs of vertice {} cannot be aggregated. Please check if the shapes are equal!".format(self.name))
        else:
            res = inputs[0]
        return res

    def clear_value(self):
        self._value = None

    def value(self, clear_value=False):
        if clear_value:
            self.clear_value()

        if self._value is None:
            self._value = self._value_function(self._aggregate_inputs([pi.value() for pi in self.parent]))
        return self._value

    def profile(self, profiler, n_runs=100):
        try:
            return profiler.profile(static_module=self, n_run=100)
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

    def __call__(self):
        raise Exception('StaticModules cannot be called! Use StaticModule.value() instead.')

#----------------------------------------some basic StaticModules--------------------------------------------

class Input(StaticModule):
    def __init__(self, value=None, *args, **kwargs):
        """
        An input op to the graph, which can store input values to propagate through the graph. If the input node has
        parent, it is the identity op, which just feeds the aggregated inputs to the output.
        """
        super(Input, self).__init__(parent=[], *args, **kwargs)
        self._value     = value
        self._shape     = self._value.shape

    def _init_parents(self, parent):
        """
        An input vertice can have no parents
        """
        pass

    def _shape_function(self, parent):
        return self._shape

    def _value_function(self, inputs):
        return self._value

    def clear_value(self):
        """
        We are not allowed to clear the value of an input vertice!
        """
        pass

class Identity(StaticModule):
    """
    The identity operation.
    """
    def _value_function(self, inputs):
        return inputs

    def _shape_function(self, parent):
        return parent[0].shape

class Merge(StaticModule):

    def _value_function(self, inputs):
        return inputs

    def _shape_function(self, parent):
        return parent[0].shape

    def _shapes_valid(self, parent):
        """
        This vertice can handle input tensors with different channel dimensions and different spatial dimensions.
        The restriction is, that the the spatial dimensions of all inputs divided by the smallest tensor dimension is a power of two,
        such that we can match the spatial dimension, using pooling.
        """
        dim_matchable = []
        for pi in parent:
            expansion_ratio = np.array(pi.shape[2:]) / np.array(self.shape[2:])
            dim_matchable.append(np.sum(np.log2(expansion_ratio)%1) == 0)
        return sum(dim_matchable) == len(dim_matchable)

    def _create_modules(self):
        """
        Take a look if we need to
        """
        shapes = [(list(pi.shape) + 4 * [1])[:4] for pi in self.parent]
        #create a list of all inputs where we need to adapt the channel dimension
        self._channel_adaptation    = [(i, self.shape[1]) for i,si in enumerate(shapes) if si[1] != self.shape[1]]
        #create a list of all channels where we need to adapt the spatial dimension
        self._shape_adaptation      = [(i, np.array(si[2:]) / np.array(self.shape[2:])) for i,si in enumerate(shapes) if tuple(si[2:]) != tuple(self.shape[2:])]

        self._channel_adaptation_modules = ModuleDict()
        for cai in self._channel_adaptation:
            ca_module = Conv(in_channels=shapes[cai[0]][1], out_channels=cai[1], kernel=(1,1))
            self._channel_adaptation_modules.add_module(cai[0], ca_module)

    def _shape_function(self, parent):
        """
        The shape of the output is the minimal shape of all input tensors.
        """
        shapes      = [(list(pi.shape) + 4 * [1])[:4] for pi in parent]
        min_shp     = tuple(np.min(np.array(shapes), axis=0))

        return min_shp

    def _aggregate_inputs(self, input):
        #1. adapt the spatial dimensions, using pooling
        for sai in self._shape_adaptation:
            input[sai[0]] =  F.average_pooling(input[sai[0]],
                                              kernel=sai[1],
                                              stride=sai[1],
                                              ignore_border=False)

        #2. adapt the channel dimension, using 1x1 convolution
        for cai in self._channel_adaptation:
            input[cai[0]] =  self._channel_adaptation_modules[cai[0]](input[cai[0]])

        print(input)

        res = 0
        #3. accumulate
        for ii in input:
            res += ii

        return res

class Join(StaticModule):
    def __init__(self, name, parent, join_parameters=None, mode='linear', *args, **kwargs):
        """
        :param join_parameters: nnabla variable of shape (#parents,1), the logits for the join operations
        :param mode: string, the mode we use to join (linear, sample, max)
        """
        if len(parent) < 2:
            raise Exception("Join vertice {} must have at least 2 inputs, but has {}.".format(self.name, len(parent)))

        self._supported_modes =['linear', 'sample', 'max']
        self.mode = mode

        if join_parameters is not None:
            if join_parameters.size == len(parent):
                self._join_parameters = F.reshape(join_parameter, shape=(len(parent),))
            else:
                raise Exception("The number of provided join parameters does not match the number of parents")
        else:
            self._join_parameters = Parameter(shape=(len(parent),))
        self._sel_p = F.softmax(self._join_parameters)

        super(Join, self).__init__(name=name, parent=parent, *args, **kwargs)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, m):
        if m in self._supported_modes:
            self._mode = m
        else:
            raise Exception("Join only supports the modes: {}".format(self._supported_modes))

    def _aggregate_inputs(self, inputs):
        """
        Aggregates all input tensors to one single input tensor (summing them up)
        """
        def one_hot(x, n=len(inputs)):
            return np.array([int(i == x) for i in range(n)])

        self._sel_p = F.softmax(self._join_parameters) # a nnabla variable
        res = 0.0
        if self.mode == 'linear':
            for pi, inpi in zip(self._sel_p, inputs):
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

    def _value_function(self, input):
        return input

#------------------------------------A graph of StaticModules--------------------------------------
class StaticGraph(StaticModule):
    # Graph is derived from Op, such that we can realize nested graphs!
    def __init__(self, name, parent=[], *args, **kwargs):
        super(StaticGraph, self).__init__(name=name, parent=parent, *args, **kwargs)
        self._graph_modules = []

        self._inputs = []
        for pi in self._parent:
            if isinstance(pi, StaticModule):
                self._inputs.append(pi)
            else:
                raise Exception("Parents of StaticGraph object must be an instance of StaticModule!")
        self._output = self.generate_graph(self._inputs)

    def _generate_graph(self, inputs):
        raise NotImplementedError

    def generate_graph(self, inputs):
        """
        This function instantiates all vertices within this graph and collects the vertices.
        """
        output = self._generate_graph(inputs)

        return output

    def clear_value(self):
        for gvi in self._graph_modules:
            gvi.clear_value()

    def value(self, clear_value=False):
        return self._output.value(clear_value=clear_value)

    @property
    def shape(self):
        """
        The output determines the shape of the graph.
        """
        if self._shape is None:
            self._shape = self._output.shape
        return self._shape

    def profile(self, profiler, n_runs=100):
        """
        Compared to a single vertice, we need to profile all vertices which are nested in this graph.
        """
        result = {}
        for mi in self.get_modules():
            if isinstance(mi[1], StaticModule) and mi[1] != self:
                result[mi[1].name] = mi[1].profile(profiler, n_runs=100)

        return result

if __name__ == '__main__':
    import nnabla as nn
    from nnabla_nas.module.profiling import NNablaProfiler

    class MyGraph(StaticGraph):
        def _generate_graph(self, inputs):
            self.input_module_2 = Input(name='input_2', value=nn.Variable((10,20,32,32)))
            self.join = Join(name='join', parent=[*inputs, self.input_module_2])
            return self.join

    input_module_1 = Input(name='input_1', value=nn.Variable((10,20,32,32)))
    myGraph = MyGraph(name='myGraph', parent=[input_module_1])

    latency = myGraph.profile(NNablaProfiler())
    import pdb; pdb.set_trace()
