from collections import OrderedDict

import nnabla as nn
import numpy as np
from nnabla.initializer import ConstantInitializer

import nnabla_nas.module as mo
import nnabla_nas.module.static.static_module as smo
from nnabla_nas.module.parameter import Parameter


#---------------------Definition of the candidate convolutions for the zoph search space-------------------------------
class SepConv(smo.SepConv):
    def __init__(self, name, parent, out_channels, kernel, dilation):

        if dilation is None:
            pad = tuple([ki//2 for ki in kernel])
        else:
            pad = tuple([(ki//2)*di for ki,di in zip(kernel, dilation)])

        smo.SepConv.__init__(self, name=name, parent=parent, in_channels=parent.shape[1],
                             out_channels=parent.shape[1],
                             kernel=kernel, pad=pad,
                             dilation=dilation, with_bias=False)

        self.bn = mo.BatchNormalization(n_features=self._out_channels, n_dims=4)
        self.relu = mo.ReLU()

    def _value_function(self, input):
        return self.relu(self.bn(smo.SepConv._value_function(self, smo.SepConv._value_function(self, input))))

class SepConv3x3(SepConv):
    def __init__(self, name, parent, channels):
        SepConv.__init__(self, name, parent, channels, (3,3), None)

class SepConv5x5(SepConv):
    def __init__(self, name, parent, channels):
        SepConv.__init__(self, name, parent, channels, (5,5), None)

class DilSepConv3x3(SepConv):
    def __init__(self, name, parent, channels):
        SepConv.__init__(self, name, parent, channels, (3,3), (2,2))

class DilSepConv5x5(SepConv):
    def __init__(self, name, parent, channels):
        SepConv.__init__(self, name, parent, channels, (5,5), (2,2))


#---------------------Definition of the candidate pooling operations for the zoph search space-------------------------------
class MaxPool3x3(smo.MaxPool):
    def __init__(self, name, parent, *args, **kwargs):
        smo.MaxPool.__init__(self, name, parent, kernel=(3,3), stride=(1,1), pad=(1,1))
        self.bn = mo.BatchNormalization(n_features=self.parent.shape[1], n_dims=4)
        self.relu= mo.ReLU()

    def _value_function(self, input, *args, **kwargs):
        return self.relu(self.bn(smo.MaxPool._value_function(self, input)))

class AveragePool3x3(smo.AvgPool):
    def __init__(self, name, parent, *args, **kwargs):
        smo.AvgPool.__init__(self, name, parent, kernel=(3,3), stride=(1,1), pad=(1,1))
        self.bn = mo.BatchNormalization(n_features=self.parent.shape[1], n_dims=4)
        self.relu= mo.ReLU()

    def _value_function(self, input):
        return self.relu(self.bn(smo.AvgPool._value_function(self, input)))

#---------------------List of candidate layers for the zoph search space-------------------------------
ZOPH_CANDIDATES = [SepConv3x3,
                   SepConv5x5,
                   DilSepConv3x3,
                   DilSepConv5x5,
                   MaxPool3x3,
                   AveragePool3x3,
                   smo.Identity,
                   smo.Zero]

#-----------------------------------The definition of a Zoph cell--------------------------------------
class ZophBlock(smo.Graph):
    def __init__(self, name, parents, candidates, channels, join_parameters=None):
        self._candidates = candidates
        self._channels = channels
        if join_parameters is None:
            self._join_parameters = Parameter(shape=(len(candidates),))
        else:
            self._join_parameters = join_parameters
        smo.Graph.__init__(self, name, parents)

        #add an input concatenation

        input_con = smo.Merging(name='{}/input_con'.format(self.name),
                                parents=self.parent,
                                mode='concat',
                                axis=1)
        self.append(input_con)
        input_conv = smo.Conv(name='{}/input_conv'.format(self.name), parent=input_con, in_channels=input_con.shape[1],
                              out_channels=self._channels, kernel=(1,1))
        self.append(input_conv)

        for i,ci in enumerate(self._candidates):
            self.append(ci(name='{}/candidate_{}'.format(self.name, i), parent=input_conv, channels=self._channels))
        self.append(smo.Join(name='{}/join'.format(self.name), parents=self[2:], join_parameters=self._join_parameters))

class ZophCell(smo.Graph):
    def __init__(self, name, parent, candidates, channels, n_modules=3, reducing=False, join_parameters=[None]*3):
        self._candidates = candidates
        self._channels = channels
        self._n_modules = n_modules
        self._reducing = reducing
        self._join_parameters = join_parameters
        smo.StaticGraph.__init__(self, name, parent)

    def _generate_graph(self, inputs):
        #match the input dimensions
        shapes = [(list(ii.shape) + 4 * [1])[:4] for ii in self.parent]
        min_shape = np.min(np.array(shapes), axis=0)
        self._shape_adaptation = {i: np.array(si[2:]) / min_shape[2:] for i,si in enumerate(shapes) if tuple(si[2:]) != tuple(min_shape[2:])}

        #perform the input channel projection, using pointwise convolutions
        projected_inputs = []
        for i,ii in enumerate(inputs):
            self.add_module(smo.StaticConv(name='{}/input_conv_{}'.format(self.name, i), parent=[ii],
                                           in_channels=ii.shape[1], out_channels=self._channels, kernel=(1,1), with_bias=False))
            projected_inputs.append(self._modules[-1])

        #perform shape adaptation, using pooling, if needed
        for i,pii in enumerate(projected_inputs):
            if i in self._shape_adaptation:
                self.add_module(smo.StaticMaxPool(name='{}/shape_adapt_pool_{}'.format(self.name, i), parent=[pii],
                                              kernel=self._shape_adaptation[i], stride=self._shape_adaptation[i]))
                projected_inputs[i] = self._modules[-1]

        #in case of a reducing cell, we need to perform another max-pooling on all inputs
        if self._reducing:
            for i, pii in enumerate(projected_inputs):
                self.add_module(smo.StaticMaxPool(name='{}/shape_adapt_pool_{}'.format(self.name, i), parent=[pii],
                                              kernel=(2,2), stride=(2,2)))
                projected_inputs[i] = self._modules[-1]

        cell_modules=projected_inputs
        for i in range(self._n_modules):
            self.add_module(ZophBlock(name='{}/zoph_block_{}'.format(self.name, i), parent=cell_modules,
                                      candidates=self._candidates, channels=self._channels,
                                      join_parameters=self._join_parameters[i]))
            cell_modules.append(self._modules[-1])

        #perform output concatenation
        self.add_module(smo.StaticDwConcatenate(name=self.name+'/output_concat', parent=cell_modules))
        return self._modules[-1]

#class ZophNetwork(smo.StaticModule):
#    def __init__(self, name, parent, stem_channels=64, cells=[ZophCell]*3, candidates=0, reducing):
#        self._stem_channels = stem_channels
#        self._cells = cells
#        self._candidates = candidates
#        self._reducing = reducing
#        smo.StaticModule.__init__(self, name, parent)
#
#    def _create_modules(self):
#        #1. add the stem convolutions
#        self.add_module()
#        #2. add the cells using shared architecture parameters
#
#        #3. add output convolution and global average pooling

if __name__ == '__main__':

    input = smo.Input(name='input', value=nn.Variable((10,20,32,32)))
    input2 = smo.Input(name='input', value=nn.Variable((10,20,32,32)))
    sep_conv3x3 = SepConv3x3(name='conv1', parent=input, channels=30)
    sep_conv5x5 = SepConv5x5(name='conv2', parent=input, channels=30)
    dil_sep_conv3x3 = DilSepConv3x3(name='conv3', parent=input, channels=30)
    dil_sep_conv5x5 = DilSepConv5x5(name='conv4', parent=input, channels=30)
    max_pool3x3 = MaxPool3x3(name='pool1', parent=input)
    avg_pool3x3 = AveragePool3x3(name='pool2', parent=input)
    zoph_block = ZophBlock(name='block1', parents=[input, input2], channels=20, candidates=ZOPH_CANDIDATES)
    #zoph_cell = ZophCell(name='cell1', parent=[input, input2], candidates=ZOPH_CANDIDATES, channels=32)

    out_1 = sep_conv3x3()
    out_2 = sep_conv5x5()
    out_3 = dil_sep_conv3x3()
    out_4 = dil_sep_conv5x5()
    out_5 = max_pool3x3()
    out_6 = avg_pool3x3()
    out_7 = zoph_block()
    import pdb; pdb.set_trace()
