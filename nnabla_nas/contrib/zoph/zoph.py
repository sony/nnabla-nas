import nnabla as nn
from collections import OrderedDict
from nnabla.initializer import ConstantInitializer
import nnabla_nas.module.static_module as smo
from nnabla_nas.module.parameter import Parameter

#---------------------Definition of the candidate layers for the zoph search space-------------------------------
#TODO: SepConvs should be applied twice, with a relu inbetween!
class SepConv3x3(smo.StaticSepConv):
    def __init__(self, name, parent, channels):
        smo.StaticSepConv.__init__(self, name=name, parent=parent, in_channels=parent[0].shape[1], out_channels=channels, kernel=(3,3), pad=(1,1))

class SepConv5x5(smo.StaticSepConv):
    def __init__(self, name, parent, channels):
        smo.StaticSepConv.__init__(self, name=name, parent=parent, in_channels=parent[0].shape[1], out_channels=channels, kernel=(5,5), pad=(2,2))

class DilSepConv3x3(smo.StaticSepConv):
    def __init__(self, name, parent, channels):
        smo.StaticSepConv.__init__(self, name=name, parent=parent, in_channels=parent[0].shape[1], out_channels=channels, kernel=(3,3),
                                            dilation=(2,2), pad=(2,2))

class DilSepConv5x5(smo.StaticSepConv):
    def __init__(self, name, parent, channels):
        smo.StaticSepConv.__init__(self, name=name, parent=parent, in_channels=parent[0].shape[1], out_channels=channels, kernel=(5,5),
                                            dilation=(2,2), pad=(4,4))

class MaxPool3x3(smo.StaticMaxPool):
    def __init__(self, name, parent, *args, **kwargs):
        super(MaxPool3x3, self).__init__(name=name, parent=parent, kernel=(3,3), pad=(1,1), stride=(1,1))

class AveragePool3x3(smo.StaticAveragePool):
    def __init__(self, name, parent, *args, **kwargs):
        super(AveragePool3x3, self).__init__(name=name, parent=parent, kernel=(3,3), pad=(1,1), stride=(1,1))

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
class ZophModule(smo.StaticGraph):
    def __init__(self, name, parent, candidates, channels, join_parameters=None):
        self._candidates = candidates
        self._channels = channels
        if join_parameters is None:
            self._join_parameters = Parameter(shape=(len(candidates),))
        else:
            self._join_parameters = join_parameters
        smo.StaticGraph.__init__(self, name, parent)

    def _generate_graph(self, inputs):
        for i,ci in enumerate(self._candidates):
            self.add_module(ci(name='{}/candidate_{}'.format(self.name, i), parent=inputs, channels=self._channels))
        self.add_module(smo.Join(name='{}/join'.format(self.name), parent=self._modules,        join_parameters=self._join_parameters))
        return self._modules[-1]



class ZophNetwork(smo.StaticModule):
    def _create_modules(self):
        pass


if __name__ == '__main__':

    input = smo.Input(name='input', value=nn.Variable((10,20,32,32)))
    sep_conv3x3 = SepConv3x3(name='conv1', parent=[input], channels=30)
    sep_conv5x5 = SepConv5x5(name='conv2', parent=[input], channels=30)
    dil_sep_conv3x3 = DilSepConv3x3(name='conv3', parent=[input], channels=30)
    dil_sep_conv5x5 = DilSepConv5x5(name='conv4', parent=[input], channels=30)
    max_pool3x3 = MaxPool3x3(name='pool1', parent=[input])
    avg_pool3x3 = AveragePool3x3(name='pool2', parent=[input])
    zoph_cell = ZophModule(name='cell1', parent=[input], channels=20)

    out_1 = sep_conv3x3()
    out_2 = sep_conv5x5()
    out_3 = dil_sep_conv3x3()
    out_4 = dil_sep_conv5x5()
    out_5 = max_pool3x3()
    out_6 = avg_pool3x3()
    out_7 = zoph_cell()
    import pdb; pdb.set_trace()
