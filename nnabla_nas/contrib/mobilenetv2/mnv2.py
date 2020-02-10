from collections import OrderedDict

import nnabla as nn

import nnabla_nas.module.static.static_module as smo
from nnabla_nas.module.parameter import Parameter

from ..model import Model
from . import modules as Mo


class ConvBnRelu6(Mo.ConvBNReLU6, smo.Module):
    def __init__(self, name, parent, *args, **kwargs):
        Mo.ConvBNReLU6.__init__(self, *args, **kwargs)
        smo.Module.__init__(self, name, parent)

    def _value_function(self, input):
        return Mo.ConvBNReLU6.call(self, input)

    def call(self, tag=None):
        return smo.Module.call(self, tag=tag)


class InvertedResidualConv(Mo.InvertedResidualConv, smo.Module):
    def __init__(self, name, parent, *args, **kwargs):
        Mo.InvertedResidualConv.__init__(self, *args, **kwargs)
        smo.Module.__init__(self, name, parent)

    def _value_function(self, input):
        return Mo.InvertedResidualConv.call(self, input)

    def call(self, tag=None):
        return smo.Module.call(self, tag=tag)


class Mnv2Classifier(smo.Graph):
    def __init__(self, name, parents, n_classes=10, drop_rate=0.2):
        super(Mnv2Classifier, self).__init__(name, parents)
        self._n_classes = n_classes
        self._drop_rate = drop_rate

        self.append(smo.Dropout(name='{}/dropout'.format(self._name),
                                parent=self.parent[0],
                                drop_prob=self._drop_rate))
        self.append(smo.GlobalAvgPool(name='{}/avg_pool'.format(self._name),
                                      parent=self[-1]))
        self.append(smo.Linear(name='{}/affine'.format(self._name),
                               parent=self[-1],
                               in_features=self[-1].shape[1],
                               out_features=self._n_classes))


class Mnv2Architecture(smo.Graph):
    def __init__(self, name, parents,
                 inverted_residual_setting,
                 first_maps=32,
                 last_maps=1280,
                 width_mult=1.0,
                 n_classes=10):
        super(Mnv2Architecture, self).__init__(name, parents)
        self._inverted_residual_setting = inverted_residual_setting
        self._n_classes = n_classes
        self._first_maps = first_maps
        self._last_maps = last_maps
        self._width_mult = width_mult
        
        # First Layer
        self.append(ConvBnRelu6(name="{}/first-conv".format(self._name),
                                    parent=self._parent[0],
                                    in_channels=self._parent[0].shape[1],
                                    out_channels=int(first_maps * width_mult),
                                    kernel=(3, 3),
                                    pad=(1, 1),
                                    stride=(2, 2),
                                    with_bias=False))
        maps = int(first_maps * width_mult)
        # Inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
                maps = int(c * width_mult)
                for i in range(n):
                    if i == 0:
                        stride = (1, 1)
                    self.append(InvertedResidualConv(name="{}/inv-resblock-{}-{}-{}-{}-{}".format(self._name,t, c, n, s, i),
                                                         parent=self[-1],
                                                         in_channels=self[-1].shape[1],
                                                         out_channels=maps,
                                                         kernel=(3, 3),
                                                         pad=(1, 1),
                                                         stride=stride,
                                                         expansion_factor=t))
        # Last Layer
        self.append(ConvBnRelu6(name="{}/last-conv".format(self._name),
                                    parent=self[-1],
                                    in_channels=maps,
                                    out_channels=int(last_maps * width_mult),
                                    kernel=(1, 1),
                                    pad=(0, 0),
                                    with_bias=False))
        
        # Classifier  
        self.append(Mnv2Classifier(name="{}/classifier".format(self._name), parents=[self[-1]], n_classes=self._n_classes))

class CandidatesCell(smo.Graph):
    def __init__(self, name, parents, t_set, stride, c, identity_skip, join_mode='linear', join_parameters=None):
        super(CandidatesCell, self).__init__(name=name, parents=parents)
        self._t_set = t_set
        self._stride = stride
        self._c = c
        self._identity_skip = identity_skip
        self._join_mode = join_mode
        if join_parameters is None:
            join_parameters = Parameter(shape=(len(t_set)+int(identity_skip),))

        for t in t_set:
            maps = c
            self.append(InvertedResidualConv(name="{}/inv-resblock-{}-{}-{}".format(self._name, t, maps, stride[0]),
                                                 parent=self._parent[0],
                                                 in_channels=self._parent[0].shape[1],
                                                 out_channels=maps,
                                                 kernel=(3, 3),
                                                 pad=(1, 1),
                                                 stride=stride,
                                                 expansion_factor=t))
        # Join
        if identity_skip:
            self.append(smo.Identity(name='{}/skip'.format(self._name), parent=self._parent[0]))
        self.append(smo.Join(name='{}/join'.format(self._name), parents=self[:], join_parameters=join_parameters, mode=self._join_mode))

class Mnv2SearchSpace(smo.Graph):
    def __init__(self, name, parents,
                 first_maps=32,
                 last_maps=1280,
                 n_classes=10,
                 *args, **kwargs):
        super(Mnv2SearchSpace, self).__init__(name=name, parents=parents, *args, **kwargs)
        self._n_classes = n_classes
        self._first_maps = first_maps
        self._last_maps = last_maps
        blocks = []
        
        #Fixed setting
        inverted_residual_setting = [
                                     #c, s
                                     [16, 1],
                                     [24, 1],
                                     [32, 1],
                                     [64, 2],
                                     [96, 1],
                                     [160, 2],
                                     [320, 1]] 

        #inverted_residual_setting = [[16, 1],[24,1]] 

        #Search space setting
        t_set = [1, 3, 6, 12] # set of expension factors
        n_max = 4 # number of inverted residual per block (can be lower because of skip connection) 

        # First Layer
        self.append(ConvBnRelu6(name="{}/first-conv".format(self._name),
                                    parent=self._parent[0],
                                    in_channels=self._parent[0].shape[1],
                                    out_channels=first_maps,
                                    kernel=(3, 3),
                                    pad=(1, 1),
                                    stride=(2, 2),
                                    with_bias=False))

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
                self.append(CandidatesCell(name="cell-{}-{}-{}".format(c,s,i),
                                            parents=[self[-1]],
                                            t_set=t_set,
                                            stride=stride,
                                            c=c,
                                            identity_skip=identity_skip))

        # Last Layer
        self.append(ConvBnRelu6(name="{}/last-conv".format(self._name),
                                    parent=self[-1],
                                    in_channels=c,
                                    out_channels=last_maps,
                                    kernel=(1, 1),
                                    pad=(0, 0),
                                    with_bias=False))
        
        # Classifier  
        self.append(Mnv2Classifier(name="{}/classifier".format(self._name), parents=[self[-1]], n_classes=self._n_classes))



class SearchNet(Model):
    r"""SearchNet for MobileNetV2."""

    def __init__(self,
                 first_maps=32,
                 last_maps=1280,
                 n_classes=10,
                 genotype = None,
                 *args, **kwargs):
        super().__init__()

        # build the network
        self.input = smo.Input(name='arch_input', value=nn.Variable((32,3,32,32)))
        self.graph = Mnv2SearchSpace(name='mnv2',
                                     parents=[self.input],
                                     first_maps=32,
                                     last_maps=1280,
                                     n_classes=10)
        # load parameters
        if genotype:
            nn.load_parameters(genotype)
            params = nn.get_parameters()
            self.set_parameters(params)

    def call(self, input):
        self.input._value = input
        return self.graph()


    def get_net_parameters(self, grad_only=False):
        param = OrderedDict()
        for key, val in self.get_parameters(grad_only).items():
            if 'join' not in key:
                param[key] = val
        return param

    def get_arch_parameters(self, grad_only=False):
        param = OrderedDict()
        for key, val in self.get_parameters(grad_only).items():
            if 'join' in key:
                param[key] = val
        return param

        
    def get_arch_modules(self):
        ans = []
        for name, module in self.get_modules():
            if isinstance(module, smo.Join):
                ans.append(module)
        return ans



if __name__ =='__main__':
    ##################### EXAMPLE MNV2 simple net ############################ 
    inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1], # NOTE: change stride 2 -> 1 for CIFAR10
            [6, 32, 3, 1],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]]

    input = smo.Input(name='arch_input', value=nn.Variable((32,3,32,32)))
    mnv2_graph = Mnv2Architecture(name='mv2',
                                  parents=[input],
                                  inverted_residual_setting=inverted_residual_setting,
                                  first_maps=32,
                                  last_maps=1280,
                                  width_mult=1.0,
                                  n_classes=10)

    print ("building nnabla graph....")
    nn_out = mnv2_graph()
    print('output shape is {}'.format(nn_out.shape))
