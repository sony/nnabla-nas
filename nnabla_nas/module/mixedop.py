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
from nnabla.initializer import ConstantInitializer
from scipy.special import softmax

from ..utils import helper
from .container import ModuleList
from .module import Module
from .parameter import Parameter
import numpy as np
import os


class MixedOp(Module):
    r"""Mixed Operator layer.

    Selects a single operator or a combination of different operators that are
    allowed in this module.

    Args:
        operators (List of `Module`): A list of modules.
        mode (str, optional): The selecting mode for this module. Defaults to
            `full`. Possible modes are `sample`, `full`, `max`, or 'fair'.
        alpha (Parameter, optional): The weights used to calculate the
            evaluation probabilities. Ignored in 'fair' mode. Defaults to None.
        rng (numpy.random.RandomState): Random generator for random choice.
        name (string): the name of this module
    """

    def __init__(self, operators, mode='full', alpha=None, rng=None, name=''):
        if mode not in ('max', 'sample', 'full', 'fair'):
            raise ValueError(f'mode={mode} is not supported.')

        Module.__init__(self, name=name)

        if alpha is None:
            n = len(operators)
            shape = (n,) + (1, 1, 1, 1)
            init = ConstantInitializer(0.0)
            alpha = Parameter(shape, initializer=init)

        if rng is None:
            rng = np.random.RandomState(313)

        self._scope_name = f'<mixedop at {hex(id(self))}>'
        self._ops = ModuleList(operators)
        self._alpha = alpha
        self._mode = mode
        self._rng = rng

        if mode == 'fair':
            n = len(operators)
            self._fair_choices = rng.choice(n, size=n, replace=False)
            self._fair_pointer = 0

    def _call_cached(self, input):
        # This method gets called when NNABLA_NAS_MIXEDOP_FAST_MODE is set.
        # See module.py init function for the call method redirections.
        output = self._train_output if self.training else self._infer_output

        if self._mode == 'full':
            if output is None:
                output = self._call_create(*args, **kwargs)

        elif self._mode in ('max', 'sample'):
            probs = softmax(self._alpha.d, axis=0)
            pvals = probs.flatten()
            index = (np.argmax(pvals) if self._mode == 'max' else
                     self._rng.choice(len(pvals), p=pvals, replace=False))
            probs[index] -= 1
            self._alpha.g = probs

            input_mask = [op.apply(need_grad=bool(i == index)).need_grad
                          for i, op in enumerate(self._ops)]

            if output is None:
                output = F.add_n(*[op(input) for op in self._ops])

            self._active = index
            output.parent.set_active_input_mask(input_mask)

        elif self._mode == 'fair':
            if self._fair_pointer == len(self._fair_choices):
                n = len(self._ops)
                self._fair_choices = self._rng.choice(n, size=n, replace=False)
                self._fair_pointer = 0
            index = self._fair_choices[self._fair_pointer]
            self._fair_pointer += 1

            input_mask = [op.apply(is_active=bool(i == index)).is_active
                          for i, op in enumerate(self._ops)]

            if output is None:
                outputs = [op(input) for op in self._ops]
                output = F.add_n(*outputs)

            self._active = index
            output.parent.set_active_input_mask(input_mask)

        if self.training:
            self._train_output = output
            return self._train_output
        else:
            self._infer_output = output
            return self._infer_output

    def call(self, input):
        if self._mode == 'full':
            h = F.stack(*[op(input) for op in self._ops], axis=0)
            h = F.mul2(h, F.softmax(self._alpha, axis=0))
            return F.sum(h, axis=0)

        if self._mode in ('max', 'sample'):
            probs = softmax(self._alpha.d, axis=0)
            pvals = probs.flatten()
            index = (np.argmax(pvals) if self._mode == 'max' else
                     self._rng.choice(len(pvals), p=pvals, replace=False))
            probs[index] -= 1
            self._alpha.g = probs

            for i, op in enumerate(self._ops):
                op.apply(need_grad=bool(i == index))

            self._active = index
            return self._ops[index](input)

        if self._mode == 'fair':
            if self._fair_pointer == len(self._fair_choices):
                n = len(self._ops)
                self._fair_choices = self._rng.choice(n, size=n, replace=False)
                self._fair_pointer = 0
            index = self._fair_choices[self._fair_pointer]
            self._fair_pointer += 1

            for i, op in enumerate(self._ops):
                op.apply(is_active=bool(i == index))

            self._active = index
            return self._ops[index](input)

    def extra_repr(self):
        return f'num_ops={len(self._ops)}, mode={self._mode}'
