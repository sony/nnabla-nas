# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nnabla as nn
import nnabla.functions as F
from nnabla.initializer import ConstantInitializer

from .module import Module
from .parameter import Parameter


class BatchNormalization(Module):
    r"""Batch normalization layer.

    Args:
        n_features (int): Number of dimentional features.
        n_dims (int): Number of dimensions.
        axes (:obj:`tuple` of :obj:`int`): Mean and variance for each element in ``axes``
            are calculated using elements on the rest axes. For example, if an input is 4 dimensions,
            and ``axes`` is ``[1]``, batch mean is calculated as ``np.mean(inp.d, axis=(0, 2, 3), keepdims=True)``
            (using numpy expression as an example).
        decay_rate (float, optional): Decay rate of running mean and
            variance. Defaults to 0.9.
        eps (float, optional): Tiny value to avoid zero division by std.
            Defaults to 1e-5.
        output_stat (bool, optional): Output batch mean and variance.
            Defaults to `False`.
        fix_parameters (bool): When set to `True`, the beta and gamma will
            not be updated.
        param_init (dict):
            Parameter initializers can be set with a dict. A key of the
            dict must be ``'beta'``, ``'gamma'``, ``'mean'`` or ``'var'``.
            A value of the dict must be an :obj:`~nnabla.initializer.
            Initializer` or a :obj:`numpy.ndarray`.
            E.g. ``{
                    'beta': ConstantIntializer(0),
                    'gamma': np.ones(gamma_shape) * 2
                    }``.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    References:
        - Ioffe and Szegedy, Batch Normalization: Accelerating Deep
        Network Training by Reducing Internal Covariate Shift.
        https://arxiv.org/abs/1502.03167
    """

    def __init__(self, n_features, n_dims, axes=[1], decay_rate=0.9, eps=1e-5,
                 output_stat=False, fix_parameters=False, param_init=None):

        assert len(axes) == 1

        shape_stat = [1 for _ in range(n_dims)]
        shape_stat[axes[0]] = n_features

        if param_init is None:
            param_init = {}
        beta_init = param_init.get('beta', ConstantInitializer(0))
        gamma_init = param_init.get('gamma', ConstantInitializer(1))
        mean_init = param_init.get('mean', ConstantInitializer(0))
        var_init = param_init.get('var', ConstantInitializer(1))

        if fix_parameters:
            self._beta = nn.Variable.from_numpy_array(
                beta_init(shape_stat))
            self._gamma = nn.Variable.from_numpy_array(
                gamma_init(shape_stat))
        else:
            self._beta = Parameter(shape_stat, initializer=beta_init)
            self._gamma = Parameter(shape_stat, initializer=gamma_init)

        self._mean = Parameter(shape_stat, need_grad=False,
                               initializer=mean_init)
        self._var = Parameter(shape_stat, need_grad=False,
                              initializer=var_init)
        self._axes = axes
        self._decay_rate = decay_rate
        self._eps = eps
        self._n_features = n_features
        self._fix_parameters = fix_parameters
        self._output_stat = output_stat

    def call(self, input):
        return F.batch_normalization(input, self._beta, self._gamma,
                                     self._mean, self._var, self._axes,
                                     self._decay_rate, self._eps,
                                     self.training, self._output_stat)

    def extra_repr(self):
        return (f'n_features={self._n_features}, '
                f'fix_parameters={self._fix_parameters}, '
                f'eps={self._eps}, '
                f'decay_rate={self._decay_rate}')
