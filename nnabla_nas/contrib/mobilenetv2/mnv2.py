import nnabla as nn
from collections import OrderedDict
from nnabla.initializer import ConstantInitializer
import nnabla_nas.module.static_module as smo
from nnabla_nas.module.parameter import Parameter


class Mnv2Classifier(StaticGraph):
    def __init__(self, name, parent, n_classes=10, drop_rate=0.2, is_training=True):
        super(Mnv2Classifier, self).__init__(name, parent)
        self._n_classes = n_classes
        self._drop_rate = drop_rate
        self._is_training = is_training

    def _generate_graph(self, inputs):
        self.add_module(StaticDropOut(name='{}/dropout'.format(self._name),
            parent=inputs,
            drop_rate=self._drop_rate,
                                      is_training=self._is_training))
        self.add_module(StaticGlobalAveragePool(name='{}/avg_pool'.format(self._name),
                                                parent=[self._modules[-1]]))
        self.add_module(StaticLinear(name='{}/affine'.format(self._name),
                               parent=[self.modules[-1]],
                               in_features=self._modules[-1].shape[1],
                               out_feature=self._n_classes))
        return self._modules[-1]


class Mnv2Architecture(StaticGraph):
    def __init__(self, name, parent,
                 inverted_residual_setting,
                 first_maps=32,
                 last_maps=1280,
                 width_mult=1.0,
                 n_classes=10,
                 is_training=True):
        super(Mnv2Architecture, self).__init__(name, parent)
        self._inverted_residual_setting = inverted_residual_setting
        self._n_classes = n_classes
        self._first_maps = first_maps
        self._last_maps = last_maps
        self._width_mult = width_mult
        self._is_training = is_training

    def _generate_graph(self, inputs):
        # First Layer
        self.add_module(ConvBnRelu6(name="{}/first-conv".format(self._name),
                                    parent=[self.modules[-1]],
                                    in_channels=inputs.shape[1],
                                    out_channels=int(first_maps * width_mult),
                                    kernel=(3, 3),
                                    pad=(1, 1),
                                    stride=(2, 2)))
        # Inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
                maps = int(c * width_mult)
                for i in range(n):
                    if i == 0:
                        stride = (s, s)
                    else:
                        stride = (1, 1)
                    self.add_module(InvertedResidualConv(name="{}/inv-resblock-{}-{}-{}-{}-{}".format(self._name,t, c, n, s, i),
                                                         parent=[self.modules[-1]],
                                                         maps=maps,
                                                         kernel=(3, 3),
                                                         pad=(1, 1),
                                                         stride=stride,
                                                         expansion_factor=t))
        # Last Layer
        self.add_module(ConvBnRelu6(name="{}/last-conv".format(self._name),
                                    parent=[self.modules[-1]],
                                    graph=self,
                                    maps=int(last_maps * width_mult),
                                    kernel=(1, 1),
                                    pad=(0, 0)))
        
        # Classifier  
        self.add_module(Mnv2Classifier(name="{}/classifier".format(self._name), parent=[self.modules[-1]], n_classes=self._n_classes))

class CandidatesCell(StaticGraph):
    def __init__(self, name, parent, t_set, stride, c, identity_skip, join_mode='linear', join_parameters=None):
        super(CandidatesCell, self).__init__(name=name, parent=parent)
        self._t_set = t_set
        self._stride = stride
        self._c = c
        self._identity_skip = identity_skip
        self._join_mode = join_mode


    def _generate_graph(self, inputs):
        for t in t_set:
            maps = c
            self.add_module(InvertedResidualConv(name="{}/inv-resblock-{}-{}-{}".format(self._name, t, maps, stride[0]),
                                                 parent=inputs,
                                                 maps=maps,
                                                 kernel=(3, 3),
                                                 pad=(1, 1),
                                                 stride=stride,
                                                 expansion_factor=t))
        # Join
        if identity_skip:
            self.add_module(Identity(name='{}/skip'.format(self._name), parent=[self.modules[-1]]))
            #TODO: deal with joinparameters 
            self.add_module(Join(name='{}/join'.format(self._name), parent=[self.modules[-1]], join_parameters=None, mode=self._join_mode))

class Mnv2SearchSpace(Graph):
    def __init__(self, name,
                 graph=None,
                 first_maps=32,
                 last_maps=1280,
                 n_classes=10,
                 *args, **kwargs):
        super(mnv2_search_space, self).__init__(name=name, graph=graph, *args, **kwargs)
        self._n_classes = n_classes
        self._first_maps = first_maps
        self._last_maps = last_maps
        blocks = []
        
        #Fixed setting

        inverted_residual_setting = [
                                #c, s
                                [16, 1],
                                [24, 1],
                                [32, 2],
                                [64, 2],
                                [96, 1],
                                [160, 2],
                                [320, 1]]

        #Search space setting
        t_set = [1, 3, 6, 12] # set of expension factors
        n_max = 4 # number of inverted residual per block (can be lower because of skip connection) 

        # First Layer
        blocks.append(ConvBnRelu6(name="first-conv",
                                  graph=self,
                                  maps=int(first_maps),
                                  kernel=(3, 3),
                                  stride=(2, 2)))
        # Inverted residual blocks
        for c,s in inverted_residual_setting:
            for i in range(n_max):
                identity_skip = True
                if i == 0:
                    stride = (s, s)
                    identity_skip = False
                else:
                    stride = (1, 1) 
                # candidates_cell
                blocks.append(candidates_cell(name="cell-{}-{}-{}".format(c,s,i),
                                              graph=self,
                                              t_set=t_set,
                                              stride=stride,
                                              c=c,
                                              identity_skip=identity_skip))
                blocks[-1](blocks[-2]) #connect it to the previous vertice

        # Last Layer
        blocks.append(ConvBnRelu6(name="last-conv",
                                  graph=self,
                                  maps=int(last_maps),
                                  kernel=(1, 1),
                                  pad=(0, 0)))
        blocks[-1](blocks[-2]) #connect it to the previous vertice
        
        # Classifier  
        blocks.append(mnv2_classifier(name="classifier", n_classes=self._n_classes))
        blocks[-1](blocks[-2]) #connect it to the previous vertice


