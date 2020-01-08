import nnabla.functions as F
from nnabla.initializer import ConstantInitializer
from scipy.special import softmax

from .. import utils as ut
from .module import Module, ModuleList
from .parameter import Parameter


class MixedOp(Module):
    def __init__(self, operators, mode='sample', alpha=None):
        super().__init__()
        n = len(operators)
        self._mode = mode
        self._mode_old = mode
        self._ops = ModuleList(operators)
        self._active = 0
        self._alpha = alpha or self._init_alpha(n)

    def __call__(self, input):
        if self._mode == 'full':
            return sum([op(input)*p for op, p in
                        zip(self._ops, F.softmax(self._alpha, axis=0))])
        self._update_active_idx()
        return self._ops[self._active](input)

    def _init_alpha(self, n):
        alpha_shape = (n,) + (1, 1, 1, 1)
        alpha_init = ConstantInitializer(0.0)
        return Parameter(alpha_shape, initializer=alpha_init)

    def _update_active_idx(self):
        """Update index of the active operation."""
        # recompute active_idx
        probs = softmax(self._alpha.d.flatten())
        self._active = ut.sample(
            pvals=probs,
            mode=self._mode
        )

        # set off need_grad for unnecessary modules
        for i, module in enumerate(self._ops):
            need_grad = self._mode == 'full' or i == self._active
            module.update_grad(need_grad)

        # update gradients for alpha
        for i, p in enumerate(probs):
            self._alpha.g[i] = int(self._active == i) - p

        return self
