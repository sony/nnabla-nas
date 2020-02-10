import nnabla as nn
import numpy as np
from nnabla.logger import logger


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
