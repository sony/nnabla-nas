from nnabla_nas.module import StaticModule
from nnabla.utils.profiler import GraphProfiler
import nnabla as nn

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

    def _profile(self, static_module, n_run):
        #1. create a dummy input to this module
        inputs = [nn.Variable(shape=pi.shape) for pi in static_module.parent]

        #2. construct the output of the static_module
        out = static_module._value_function(static_module._aggregate_inputs(inputs))

        #3. run the nnabla profiler
        prof = GraphProfiler(out, device_id=self._dev_idx, ext_name=self._nnabla_context, n_run=n_run)
        prof.run()
        return prof.result['forward_all']
