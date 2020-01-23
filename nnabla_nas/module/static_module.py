"""
Classes for graph based definition of DNN architectures and candidate spaces.
"""

#from nnabla_nas.graph.profiling import *
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla_nas.module import Module, ModuleList
from nnabla_nas.module.module import ModuleDict
from nnabla_nas.module.convolution import Conv
from nnabla_nas.module.batchnorm import BatchNormalization
from nnabla_nas.module.relu import ReLU
from nnabla_nas.module.parameter import Parameter
from nnabla.initializer import (ConstantInitializer, UniformInitializer,
                                calc_uniform_lim_glorot)

class StaticModule(Module):
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
        self._parent        = []
        self._init_parents(parent)

        self._child         = []
        self._name          = name
        self._profiler      = None
        self._value         = None
        self._eval_probs    = None
        self._eval_prob     = None
        self._shape         = None

        if not self._shapes_valid(self._parent):
            raise Exception("Input shapes of vertice {} are not valid!".format(self.name))

        self._create_modules()
        Module.__init__(self)

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
        #we build a nummy nnabla graph to infer the shapes
        inputs       = [nn.Variable(pi.shape) for pi in parent]
        dummy_graph  = self._value_function(self._aggregate_inputs(inputs))
        return dummy_graph.shape

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
        #import pdb; pdb.set_trace()
        return self._parent

    @property
    def child(self):
        """
        The vhild vertices of this vertice (all vertices which are connected
        with outgoing edges).
        """
        return self._child

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

    def _value_function(self, input):
        raise NotImplementedError

    def clear_value(self):
        self._value = None

    def __call__(self, clear_value=False):
        if clear_value:
            self.clear_value()

        if self._value is None:
            self._value = self._value_function(self._aggregate_inputs([pi(clear_value) for pi in self.parent]))
        return self._value

    def clear_eval_probs(self):
        self._eval_probs = None

    def eval_probs(self, clear_probs=False):
        if clear_probs:
            self.clear_eval_probs()

        if self._eval_probs is None:
            self._eval_probs = self._eval_prob_function([pi.eval_probs(clear_probs) for pi in self.parent])
        return self._eval_probs

    def _eval_prob_function(self, input):
        """
        The standard transformation is, that we multiply with a path probability of 1.0
        (deterministic graph with independent (Bernoulli distributed) path probabilities).
        :param input: list of dictionaries with evaluation probabilities of the parent vertices
        :param name_scope: string
        :return: updated evaluation probabilities
        """
        res     = {}
        for inpi in input: #dictionary
            for inpii in inpi:
                if inpii in res:
                    res[inpii].append(inpi[inpii])
                else:
                    res.update({inpii: [inpi[inpii]]})

        #combine all probabilities arriving from different paths
        for resi in res:
            res[resi] = 1.0 - F.prod(1.0 - F.stack(*res[resi]))

        res.update({self: nn.Variable.from_numpy_array(np.array(1.0))})
        return res

    def profile(self, profiler, n_run=100):
        try:
            return profiler.profile(static_module=self, n_run=n_run)
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

class Zero(StaticModule):
    def _value_function(self, input):
        return 0.0*input

    def _eval_prob_function(self, inputs): #a zero operation sets the evaluation pobabilities of all parents to 0
        return {self: nn.Variable.from_numpy_array(np.array(1.0))}

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
    def __init__(self, name, parent, join_parameters, mode='linear', *args, **kwargs):
        """
        :param join_parameters: nnabla variable of shape (#parents,1), the logits for the join operations
        :param mode: string, the mode we use to join (linear, sample, max)
        """
        if len(parent) < 2:
            raise Exception("Join vertice {} must have at least 2 inputs, but has {}.".format(self.name, len(parent)))

        self._supported_modes =['linear', 'sample', 'max']
        self.mode = mode

        if join_parameters.size == len(parent):
            self._join_parameters = F.reshape(join_parameters, shape=(len(parent),))
        else:
            raise Exception("The number of provided join parameters does not match the number of parents")
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

    def _eval_prob_function(self, input):
        """
        In case of JoinLin, we multiply with a path probability self._sel_p
        (Categorical path probabilities).
        :param input: list of dictionaries with evaluation probabilities of the parent vertices
        :param name_scope: string
        :return: updated evaluation probabilities
        """
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


#------------------------------------Some pooling StaticModules------------------------------------
class MaxPool(Module):
    def __init__(self, kernel, stride=None, ignore_border=True, pad=None, channel_last=False, n_outputs=-1, outputs=None):
        self._kernel = kernel
        self._stride = stride
        self._ignore_border=ignore_border
        self._pad = pad
        self._channel_last = channel_last
        self._n_outputs = n_outputs
        self._outputs = outputs

    def __call__(self, input):
        return F.max_pooling(input, self._kernel, self._stride, self._ignore_border, self._pad, self._channel_last, self._n_outputs, self._outputs)

class AveragePool(Module):

    def __init__(self, kernel, stride=None, ignore_border=True, pad=None, including_pad=True, channel_last=False, n_outputs=-1, outputs=None):
        self._kernel = kernel
        self._stride = stride
        self._ignore_border=ignore_border
        self._pad = pad
        self._including_pad = including_pad
        self._channel_last = channel_last
        self._n_outputs = n_outputs
        self._outputs = outputs

    def __call__(self, input):
        return F.average_pooling(input, self._kernel, self._stride, self._ignore_border, self._pad, self._including_pad, self._channel_last, self._n_outputs, self._outputs)

class GlobalAveragePool(Module):
    def __init__(self, n_outputs=-1, outputs=None):
        self._n_outputs = n_outputs
        self._outputs = outputs

    def __call__(self, input):
        return F.global_average_pooling(input, self._n_outputs, self._outputs)

#------------------------------------Some Modules needed to define static modules------------------------------

class DwConv(Module):
    def __init__(self, in_channels, kernel,
                pad=None, stride=None, dilation=None,
                w_init=None, b_init=None, rng=None, with_bias=True):
        Module.__init__(self)
        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(
                    in_channels,
                    in_channels,
                    tuple(kernel)),
                rng=rng)
        if with_bias and b_init is None:
            b_init = ConstantInitializer()
        self.W = Parameter((in_channels,) + tuple(kernel), initializer = w_init)
        self.b = None
        if with_bias:
            b_init = ConstantInitializer()
        self.b = Parameter((in_channels, ), initializer=b_init)
        self.pad = pad
        self.stride = stride
        self.dilation=dilation
        self.in_channels=in_channels
        self.rng=rng
        self.with_bias=with_bias

    def __call__(self, input):
        return F.depthwise_convolution(input, self.W, self.b, 1,
                            self.pad, self.stride, self.dilation)

class SepConv(DwConv):
    def __init__(self, out_channels, *args, **kwargs):
        DwConv.__init__(self, *args, **kwargs)
        self.out_channels = out_channels
        self._conv_module_pw = Conv(self.in_channels, self.out_channels, kernel=(1, 1),
                                 pad=None, group=1, rng=self.rng, with_bias=None)

    def __call__(self, input):
        return self._conv_module_pw(DwConv.__call__(self, input))

#------------------------------------Some convolutional StaticModules------------------------------
class StaticDwConcatenate(StaticModule):
    def _aggregate_inputs(self, inputs):
        if len(inputs)>1:
            return F.concatenate(*inputs, axis=1)
        else:
            return inputs[0]

    def _value_function(self, input):
        return input

class StaticConv(Conv, StaticModule):
    def __init__(self, name, parent, *args, **kwargs):
        Conv.__init__(self, *args, **kwargs)
        StaticModule.__init__(self, name, parent)

    def _value_function(self, input):
        return Conv.__call__(self, input) #change to self.call

    def __call__(self, clear_value=False):
        return StaticModule.__call__(self, clear_value=clear_value)

class StaticSepConv(SepConv, StaticModule):
    def __init__(self, name, parent, *args, **kwargs):
        SepConv.__init__(self, *args, **kwargs)
        StaticModule.__init__(self, name, parent)

    def _value_function(self, input):
        return SepConv.__call__(self, input)

    def __call__(self, clear_value=False):
        return StaticModule.__call__(self, clear_value=clear_value)

class StaticMaxPool(MaxPool, StaticModule):
    def __init__(self, name, parent, *args, **kwargs):
        MaxPool.__init__(self, *args, **kwargs)
        StaticModule.__init__(self, name, parent)

    def _value_function(self, input):
        return MaxPool.__call__(self, input)

    def __call__(self, clear_value=False):
        return StaticModule.__call__(self, clear_value=clear_value)

class StaticAveragePool(AveragePool, StaticModule):
    def __init__(self, name, parent, *args, **kwargs):
        AveragePool.__init__(self, *args, **kwargs)
        StaticModule.__init__(self, name, parent)

    def _value_function(self, input):
        return AveragePool.__call__(self, input)

    def __call__(self, clear_value=False):
        return StaticModule.__call__(self, clear_value=clear_value)

class StaticGlobalAveragePool(GlobalAveragePool, StaticModule):
    def __init__(self, name, parent, *args, **kwargs):
        GlobalAveragePool.__init__(self, *args, **kwargs)
        StaticModule.__init__(self, name, parent)

    def _value_function(self, input):
        return GlobalAveragePool.__call__(self, input)

    def __call__(self, clear_value=False):
        return StaticModule.__call__(self, clear_value=clear_value)

class StaticReLU(ReLU, StaticModule):
    def __init__(self, name, parent, *args, **kwargs):
        ReLU.__init__(self, *args, **kwargs)
        StaticModule.__init__(self, name, parent)

    def _value_function(self, input):
        return ReLU.__call__(self, input)

    def __call__(self, clear_value=False):
        return StaticModule.__call__(self, clear_value=clear_value)

class StaticBatchNormalization(BatchNormalization, StaticModule):
    def __init__(self, name, parent, *args, **kwargs):
        BatchNormalization.__init__(self, *args, **kwargs)
        StaticModule.__init__(self, name, parent)

    def _value_function(self, input):
        return BatchNormalization.__call__(self, input)

    def __call__(self, clear_value=False):
        return StaticModule.__call__(self, clear_value=clear_value)


#------------------------------------A graph of StaticModules--------------------------------------
class StaticGraph(ModuleList, StaticModule): #TODO: change to Sequential
    # Graph is derived from Op, such that we can realize nested graphs!
    def __init__(self, name, parent=[], *args, **kwargs):
        ModuleList.__init__(self)
        StaticModule.__init__(self, name=name, parent=parent, *args, **kwargs)

        self._inputs = [] #rename to parent
        for pi in self._parent:
            if isinstance(pi, StaticModule):
                self._inputs.append(pi)
            else:
                raise Exception("Parents of StaticGraph object must be an instance of StaticModule!")
        self._output = self.generate_graph(self._inputs)

    def _generate_graph(self, inputs): #remove and use append/extend
        raise NotImplementedError

    def generate_graph(self, inputs): #remove
        """
        This function instantiates all vertices within this graph and collects the vertices.
        """
        output = self._generate_graph(inputs)
        return output

    def clear_value(self):
        for gmi in self._graph_modules:
            gmi.clear_value()

    def clear_probs(self):
        for gmi in self._graph_modules:
            gmi.clear_value()

    def __call__(self, clear_value=False):
        return self._output(clear_value=clear_value)

    def eval_probs(self, clear_probs=False):
        return self._output.eval_probs(clear_probs=clear_probs)

    @property
    def shape(self):
        """
        The output determines the shape of the graph.
        """
        if self._shape is None:
            self._shape = self._output.shape
        return self._shape

    def profile(self, profiler, n_run=100):
        """
        Compared to a single vertice, we need to profile all vertices which are nested in this graph.
        """
        result = {}
        for mi in self.get_modules():
            if isinstance(mi[1], StaticModule) and mi[1] != self:
                result[mi[1].name] = mi[1].profile(profiler, n_run=n_run)

        return result

if __name__ == '__main__':
    import nnabla as nn
    from nnabla_nas.module.profiling import NNablaProfiler

    class MyGraph(StaticGraph):
        def _generate_graph(self, inputs):
            self.add_module(Input(name='input_2', value=nn.Variable((10,20,32,32)))) #change add_module -> append
            self.add_module(StaticSepConv(name='conv', parent=[self._modules[-1]], in_channels=20, out_channels=20, kernel=(3,3), pad=(1,1)))
            self.add_module(Join(name='join', parent=[*inputs, self._modules[-1]], join_parameters=Parameter(shape=(2,))))
            self.add_module(Merge(name='merge', parent=[self._modules[-1], self._modules[0]]))
            return self._modules[-1]

    input_module_1 = Input(name='input_1', value=nn.Variable((10,20,32,32)))
    myGraph = MyGraph(name='myGraph', parent=[input_module_1])
    out = myGraph()

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
