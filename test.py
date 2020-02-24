
from nnabla_nas.utils import CommunicatorWrapper
from nnabla_ext.cuda import StreamEventHandler

import nnabla as nn
from nnabla.ext_utils import get_extension_context

import numpy as np

if __name__ == "__main__":
    # setup context for nnabla
    device_id = 0
    ctx = get_extension_context('cudnn', device_id=0)
    nn.set_default_context(ctx)
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)

    from nnabla_nas.contrib.mbn import SearchNet

    net = SearchNet(10)
    rng = np.random.RandomState(1234)
    params = net.get_net_parameters()
    key = list(params.keys())[100]
    s = str(params[key].d.flatten()[:10])
    print(f'{comm.rank} = {s}, {comm.n_procs}')

    # print(params['_features/22/_mixed/_ops/4/_conv/1/1/_var'].d)

    stream_event_handler = StreamEventHandler(int(comm.ctx.device_id))

    print(stream_event_handler)
    # v = 0
    # r = comm.rank
    # acc = nn.NdArray((1,))
    # acc.zero()
    # acc.data = r

    # stream_event_handler.event_synchronize()
    # comm.all_reduce(acc,  division=True, inplace=False)
    # stream_event_handler.add_default_stream_event()
    # print(f'c={comm.rank}, acc={acc.data}, ')

    # rng = np.random.RandomState(42)
    # ran = rng.choice(5, p=np.ones((5, ))/5, replace=False)
    # print(
    #     f'communicator={comm.rank}, v = {v}, rand={ran} device_id={device_id}, process={comm.n_procs}')
    # ran = rng.choice(5, p=np.ones((5, ))/5, replace=False)
    # print(
    #     f'communicator={comm.rank}, v = {v}, rand={ran} device_id={device_id}, process={comm.n_procs}')
