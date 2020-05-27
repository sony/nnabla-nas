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

import nnabla.functions as F
from nnabla import random
from nnabla.initializer import ConstantInitializer
from scipy.special import softmax

from ..utils import helper
from .container import ModuleList
from .module import Module
from .parameter import Parameter


class MixedOp(Module):
    r"""Mixed Operator layer.

    Selects a single operator or a combination of different operators that are
    allowed in this module.

    Args:
        operators (List of `Module`): A list of modules.
        mode (str, optional): The selecting mode for this module. Defaults to
            `full`. Possible modes are `sample`, `full`, or `max`.
        alpha (Parameter, optional): The weights used to calculate the
            evaluation probabilities. Defaults to None.
        rng (numpy.random.RandomState): Random generator for random choice.
    """

    def __init__(self, operators, mode='full', alpha=None, rng=None):
        if mode not in ('max', 'sample', 'full'):
            raise ValueError(f'mode={mode} is not supported.')

        self._active = None  # save the active index
        self._mode = mode
        self._ops = ModuleList(operators)
        self._alpha = alpha
        if rng is None:
            rng = random.prng
        self._rng = rng

        if alpha is None:
            n = len(operators)
            shape = (n,) + (1, 1, 1, 1)
            init = ConstantInitializer(0.0)
            self._alpha = Parameter(shape, initializer=init)

    def call(self, input):
        if self._mode == 'full':
            out = F.stack(*[op(input) for op in self._ops], axis=0)
            out = F.mul2(out, F.softmax(self._alpha, axis=0))
            return F.sum(out, axis=0)

        # update active index
        self._update_active_index()

        return self._ops[self._active](input)

    def _update_active_index(self):
        """Update index of the active operation."""
        probs = softmax(self._alpha.d, axis=0)
        self._active = helper.sample(
            pvals=probs.flatten(),
            mode=self._mode,
            rng=self._rng
        )
        # update gradients
        probs[self._active] -= 1
        self._alpha.g = probs

        for i, op in enumerate(self._ops):
            op.apply(need_grad=(self._active == i))

    def extra_repr(self):
        return f'num_ops={len(self._ops)}, mode={self._mode}'
