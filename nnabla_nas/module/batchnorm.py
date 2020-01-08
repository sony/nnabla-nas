import nnabla.functions as F
from nnabla.initializer import ConstantInitializer

from .module import Module
from .parameter import Parameter


class BatchNormalization(Module):
    def __init__(self, n_features, n_dims, axes=[1], decay_rate=0.9, eps=1e-5,
                 batch_stat=True, output_stat=False, fix_parameters=False,
                 param_init=None):
        assert len(axes) == 1
        shape_stat = [1 for _ in range(n_dims)]
        shape_stat[axes[0]] = n_features

        if param_init is None:
            param_init = {}
        beta_init = param_init.get('beta', ConstantInitializer(0))
        gamma_init = param_init.get('gamma', ConstantInitializer(1))
        mean_init = param_init.get('mean', ConstantInitializer(0))
        var_init = param_init.get('var', ConstantInitializer(1))

        super().__init__()

        self.beta = Parameter(shape_stat, initializer=beta_init)
        self.gamma = Parameter(shape_stat, initializer=gamma_init)
        self.mean = Parameter(shape_stat, initializer=mean_init)
        self.var = Parameter(shape_stat, initializer=var_init)
        self.axes = axes
        self.decay_rate = decay_rate
        self.eps = eps
        self.output_stat = output_stat

    def __call__(self, input):
        return F.batch_normalization(input, self.beta, self.gamma,
                                     self.mean, self.var, self.axes,
                                     self.decay_rate, self.eps, self.training,
                                     self.output_stat)
