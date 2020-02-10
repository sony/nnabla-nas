import nnabla as nn

import nnabla_nas.module.static.static_module as smo
from mnv2 import SearchNet

# build the network
inp = smo.Input(name='arch_input', value=nn.Variable((32,3,32,32)))
graph = SearchNet(name='mnv2',
                 first_maps=32,
                 last_maps=1280,
                 n_classes=10)


arch = [0,4,4,4,2,2,4,4,2,2,2,4,2,2,2,2,2,2,2,4,2,2,2,4,2,4,4,4] 

t_set = [1,3,6,12,0]

i=0
for k,v in graph.get_arch_parameters().items():
    print("{} : residual block with t= {}".format(k, t_set[arch[i]]))
    v.d[arch[i]] = 1.0
    i+=1


nn.save_parameters('reference_arch.h5', graph.get_arch_parameters())
