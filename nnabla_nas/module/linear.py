import nnabla.functions as F
from nnabla.initializer import (ConstantInitializer, UniformInitializer,
                                calc_uniform_lim_glorot)

from .module import Module
from .parameter import Parameter


class Linear(Module):

    def __init__(self, in_features, out_features,
                 base_axis=1, w_init=None, b_init=None,
                 rng=None, bias=True):
        super().__init__()
        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(in_features, out_features), rng=rng)
        self.W = Parameter((in_features, out_features), initializer=w_init)
        self.b = None
        if bias:
            if b_init is None:
                b_init = ConstantInitializer()
            self.b = Parameter((out_features, ), initializer=b_init)
        self.base_axis = base_axis

    def __call__(self, input):
        return F.affine(input, self.W, self.b, self.base_axis)
