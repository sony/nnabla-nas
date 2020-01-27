import nnabla as nn
import numpy as np
from nnabla.utils.profiler import GraphProfiler

from ...module import Identity, Zero


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

    def __init__(self, device_id=0, ext_name='cpu', n_run=10):
        self._device_id = device_id
        self._ext_name = ext_name
        self._n_run = n_run

    def predict(self, module):
        idm = id(module)
        if idm not in self.memo:
            self.memo[idm] = dict()
        mem = self.memo[idm]
        key = '-'.join([str(k) for k in module.inputs])

        if key not in mem:
            if isinstance(module, (Identity, Zero)):
                return 0
            state = module.training
            module.apply(training=False)  # turn off training
            # run profiler
            nnabla_vars = [nn.Variable(s) for s in module.inputs]
            runner = GraphProfiler(module.call(*nnabla_vars),
                                   device_id=self._device_id,
                                   ext_name=self._ext_name,
                                   n_run=self._n_run)
            runner.time_profiling_forward()
            mem[key] = float(runner.result['forward'][0].mean_time)
            module.apply(training=state)  # recover training state

        return mem[key]
