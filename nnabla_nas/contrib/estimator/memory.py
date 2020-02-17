from .estimator import Estimator
import numpy as np


class MemoryEstimator(Estimator):
    """Estimator for the memory used."""

    def predict(self, module):
        idm = id(module)
        if idm not in self.memo:
            self.memo[idm] = sum(np.prod(p.shape)
                                 for p in module.parameters.values())
        return self.memo[idm]
