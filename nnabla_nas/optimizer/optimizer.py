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

from collections import OrderedDict

import nnabla.solvers as S


class Optimizer(object):
    r"""An Optimizer class.

    Args:
        retain_state (bool, optional): Whether retaining states is true.
            Defaults to False.
        weight_decay (float, optional): Weight decay (L2 penalty). Should be
            a positive value. Defaults to None.
        grad_clip (float, optional): An input scalar of float value. Should be
            a positive value. Defaults to None.
        lr_scheduler (`BaseLearningRateScheduler`, optional): Learning rate
            scheduler. Defaults to None (no learning rate scheduler is applied).
        name (str, optional): Name of the solver. Defaults to 'Sgd'.

    Raises:
        NotImplementedError: If the solver is not supported in NNabla.
    """

    def __init__(self,
                 retain_state=False,
                 weight_decay=None,
                 grad_clip=None,
                 lr_scheduler=None,
                 name='Sgd', **kargs):

        if name not in S.__dict__:
            raise NotImplementedError(name + 'is not implemented')
        if retain_state:
            self._states = OrderedDict()

        self._solver = S.__dict__[name](**kargs)
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        self._retain_state = retain_state
        self._lr_scheduler = lr_scheduler
        self._iter = 0  # current iter

    def set_parameters(self, params, **kargs):
        r"""Set parameters by dictionary of keys and parameter Variables."""
        if self._retain_state:
            self._states.update(self._solver.get_states())
        self._solver.set_parameters(params, **kargs)
        if self._retain_state:
            self._solver.set_states(
                OrderedDict({
                    k: v for k, v in self._states.items() if k in params
                })
            )

    def update(self):
        r"""Update parameters.

        When this function is called, parameter values are updated using the gradients
        accumulated in backpropagation, stored in the grad field of the parameter Variables.
        """
        if self._lr_scheduler is not None:
            lr = self.get_learning_rate()
            self._solver.set_learning_rate(lr)

        if self._grad_clip is not None:
            self._solver.clip_grad_by_norm(self._grad_clip)

        if self._weight_decay is not None:
            self._solver.weight_decay(self._weight_decay)

        self._solver.update()
        self._iter += 1

    def zero_grad(self):
        r"""Initialize gradients of all registered parameter by zero."""
        self._solver.zero_grad()

    def get_parameters(self):
        r"""Get all registered parameters."""
        return self._solver.get_parameters()

    def get_learning_rate(self):
        r"""Get the learning rate."""
        if self._lr_scheduler is None:
            return self._solver.learning_rate()
        return self._lr_scheduler.get_learning_rate(self._iter)

    def clear_parameters(self):
        r"""Clear all parameters."""
        self._solver.clear_parameters()
        self._iter = 0
        if self._retain_state:
            self._states.clear()
