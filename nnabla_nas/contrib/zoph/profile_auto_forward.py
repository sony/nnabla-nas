import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla_nas.contrib.zoph.zoph import ZophNetwork
import nnabla_nas.module.static.static_module as smo
import time

if __name__ == '__main__':

    ctx = nn.ext_utils.get_extension_context('cudnn', device_id='0')
    nn.set_default_context(ctx)

    input = smo.Input(name='input', value=nn.Variable((64,3,32,32)))
    zoph_network = ZophNetwork(name='network1', parents=[input])
    network_modules = zoph_network.get_modules()
    for _,mi in network_modules:
        if isinstance(mi, smo.Join):
            print(mi.name)
            mi.mode = 'linear'
    
    input_d = np.random.randn(64, 3, 32, 32)
    _ = zoph_network()

    network_modules = zoph_network.get_modules()
    for _,mi in network_modules:
        if isinstance(mi, smo.Join):
            print(mi.name)
            mi.mode = 'sample'

    def forward_static():
        nn.set_auto_forward(False)
        start = time.time()
        for i in range(100):
            input.d = input_d
            out_9 = zoph_network()
            out_9.forward()
            out_9.backward()
        _ = out_9.d
        return time.time() - start


    def forward_dynamic():
        nn.set_auto_forward(True)

        start = time.time()

        for i in range(100):
            input.d = input_d
            out_9 = zoph_network()
            out_9.backward()
        _ = out_9.d

        return time.time() - start

    print("Without auto-forward, inference of 100 different networks took {}".format(forward_static()))
    #print("With auto-forward, inference of 100 different networks took {}".format(forward_dynamic()))


