import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.initializer import ConstantInitializer
from scipy.special import softmax

from .. import utils as ut
from .module import Module, ModuleList
from .parameter import Parameter


class MixedOp(Module):
    def __init__(self, operators, mode='sample', alpha=None):
        super().__init__()
        n = len(operators)
        alpha_shape = (n,) + (1, 1, 1, 1)
        alpha_init = ConstantInitializer(0.0)

        self._mode = mode

        self._ops = ModuleList(operators)
        self._alpha = alpha or Parameter(alpha_shape, initializer=alpha_init)
        self._binary = nn.Variable.from_numpy_array(np.zeros(alpha_shape))

        self._active = 0  # save the active index
        self._state = None  # save the states of intermediate outputs

    def __call__(self, input):
        if self._mode == 'full':
            out = F.mul2(self._ops(input), F.softmax(self._alpha, axis=0))
            return F.sum(out, axis=0)

        if self._mode == 'sample':
            self._state = [op(input) for op in self._ops]
            out = F.mul2(F.stack(*self._state, axis=0), self._binary)
            return F.sum(out, axis=0)

        return self._ops[self._active](input)

    def _update_active_idx(self):
        """Update index of the active operation."""
        # recompute active_idx
        probs = softmax(self._alpha.d.flat)
        self._active = ut.sample(
            pvals=probs,
            mode=self._mode
        )
        # update binary variable
        self._binary.data.zero()
        self._binary.d[self._active] = 1

        # set off need_grad for unnecessary modules
        if self._state:
            for i in range(len(self._state)):
                self._state[i].need_grad = (i == self._active)

        return self

    def _update_alpha_grad(self):
        """Update gradients for alpha."""
        probs = softmax(self._alpha.d.flat)
        self._active = ut.sample(
            pvals=probs,
            mode=self._mode
        )
        probs[self._active] -= 1
        self._alpha.g = np.reshape(-probs, self._alpha.shape)
        return self
