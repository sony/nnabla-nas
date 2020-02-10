import nnabla as nn
from nnabla.utils.profiler import GraphProfiler

from nnabla_nas.module import Module


class Profiler(object):
    def _profile(self, static_module, n_run):
        raise NotImplementedError

    def profile(self, static_module, n_run=100):
        return self._profile(static_module, n_run)

class NNablaProfiler(Profiler):
    def __init__(self):
        ctx = nn.context.get_current_context()
        self._dev_idx = int(ctx.device_id)
        self._nnabla_context = ctx.backend[0].split(':')[0]

    def _profile(self, input, n_run):
        for i in range(2):
            input.forward() #warmups
        prof = GraphProfiler(input, device_id=self._dev_idx, ext_name=self._nnabla_context, n_run=n_run)
        prof.run()
        return float(prof.result['forward_all'])
