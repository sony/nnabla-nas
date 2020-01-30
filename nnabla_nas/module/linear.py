import nnabla.functions as F
from nnabla.initializer import (ConstantInitializer, UniformInitializer,
                                calc_uniform_lim_glorot)

from .module import Module
from .parameter import Parameter


class Linear(Module):
    r"""Linear layer.
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features (int):
        in_features (int):
        base_axis (int, optional): Dimensions up to `base_axis` are treated as
            the sample dimensions. Defaults to 1.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or
            :obj:`numpy.ndarray`): Initializer for weight. By default, it is
            initialized with :obj:`nnabla.initializer.UniformInitializer`
            within the range determined by
            :obj:`nnabla.initializer.calc_uniform_lim_glorot`.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or
            :obj:`numpy.ndarray`): Initializer for bias. By default, it is
            initialized with zeros if `with_bias` is `True`.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    """

    def __init__(self, in_features, out_features, base_axis=1, w_init=None,
                 b_init=None, rng=None, bias=True):
        Module.__init__(self)

        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(in_features, out_features), rng=rng)
        self._W = Parameter((in_features, out_features), initializer=w_init)
        self._b = None

        if bias:
            if b_init is None:
                b_init = ConstantInitializer()
            self._b = Parameter((out_features, ), initializer=b_init)

        self._base_axis = base_axis
        self._in_features = in_features
        self._out_features = out_features

    def call(self, input):
        return F.affine(input, self._W, self._b, self._base_axis)

    def __extra_repr__(self):
        return (f'in_features={self._in_features}, '
                f'out_features={self._out_features}, '
                f'bias={self._b is not None}')
