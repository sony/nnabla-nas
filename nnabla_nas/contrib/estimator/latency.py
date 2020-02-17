
import nnabla as nn
from nnabla.logger import logger
from nnabla.utils.profiler import GraphProfiler

from .estimator import Estimator


class LatencyEstimator(Estimator):
    """Latency estimator.

    Args:
        device_id (int): gpu device id.
        ext_name (str): Extension name. e.g. 'cpu', 'cuda', 'cudnn' etc.
        n_run (int): This argument specifies how many times the each functions
            execution time are measured. Default value is 10.
    """

    def __init__(self, device_id=None, ext_name=None, n_run=10):
        ctx = nn.context.get_current_context()
        if device_id is None:
            device_id = int(ctx.device_id)
        if ext_name is None:
            ext_name = ctx.backend[0].split(':')[0]
        self._device_id = device_id
        self._ext_name = ext_name
        self._n_run = n_run

    def predict(self, module):
        idm = str(module)
        if idm not in self.memo:
            self.memo[idm] = dict()
        mem = self.memo[idm]
        key = '-'.join([str(k[1:]) for k in module.input_shapes])

        if key not in mem:
            state = module.training
            module.apply(training=False)  # turn off training
            try:
                # run profiler
                nnabla_vars = [nn.Variable((1,) + s[1:])
                               for s in module.input_shapes]
                runner = GraphProfiler(module.call(*nnabla_vars),
                                       device_id=self._device_id,
                                       ext_name=self._ext_name,
                                       n_run=self._n_run)
                runner.run()
                latency = float(runner.result['forward_all'])
            except:  # noqa
                #import pdb; pdb.set_trace()
                latency = 0
                logger.warning(f'Latency calculation fails: {idm}[{key}]')

            mem[key] = latency
            module.apply(training=state)  # recover training state

        return mem[key]
