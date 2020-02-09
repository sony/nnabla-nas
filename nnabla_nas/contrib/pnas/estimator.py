import nnabla as nn
import numpy as np
from nnabla.utils.profiler import GraphProfiler
from nnabla.logger import logger
from ...module import Identity
from ...module import Zero


class Estimator(object):
    """Estimator base class."""
    @property
    def memo(self):
        if '_memo' not in self.__dict__:
            self._memo = dict()
        return self._memo

    def get_estimation(self, module):
        """Returns the estimation of the whole module."""
        return sum(self.predict(m) for _, m in module.get_modules()
                   if len(m.modules) == 0 and m.need_grad)

    def reset(self):
        """Clear cache."""
        self.memo.clear()

    def predict(self, module):
        """Predicts the estimation for a module."""
        raise NotImplementedError


class MemoryEstimator(Estimator):
    """Estimator for the memory used."""

    def predict(self, module):
        idm = id(module)
        if idm not in self.memo:
            self.memo[idm] = sum(np.prod(p.shape)
                                 for p in module.parameters.values())
        return self.memo[idm]


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
                latency = 0
                logger.warning(f'Latency calculation fails: {idm}[{key}]')

            mem[key] = latency
            module.apply(training=state)  # recover training state

        return mem[key]
