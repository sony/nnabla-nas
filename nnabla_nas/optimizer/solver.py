from collections import OrderedDict

import nnabla.solvers as S


class Solver(object):
    """Manager of the solvers"""

    def __init__(self, optim, lr, **kargs):
        self._states = dict()
        self._solver = S.__dict__[optim](lr, **kargs)
        self._solver_init = S.__dict__[optim](lr, **kargs)

    def set_parameters(self, params):
        params = self._clean_params(params)
        states = self._get_states(params)
        self._solver.set_parameters(
            params, reset=True, retain_state=False
        )
        self._solver.set_states(states)

    def weight_decay(self, wd):
        self._solver.weight_decay(wd)

    def update(self):
        self._solver.update()

    def zero_grad(self):
        self._solver.zero_grad()

    def get_parameters(self):
        return self._solver.get_parameters()

    def set_learning_rate(self, lr):
        return self._solver.set_learning_rate(lr)

    def _register_new_params(self, params):
        """Register new params"""
        if len(params):
            self._solver_init.set_parameters(
                params, reset=True, retain_state=False
            )
            self._states.update(self._solver_init.get_states())

    def _get_states(self, params):
        states = OrderedDict()
        unregistered = OrderedDict()

        for k, v in params.items():
            if k not in self._states:
                unregistered[k] = v

        self._register_new_params(unregistered)
        for k in params:
            states[k] = self._states[k]

        return states

    def _clean_params(self, params):
        clean, mark = OrderedDict(), set()
        for k, p in params.items():
            if p not in mark:
                mark.add(p)
                clean[k] = p
        return clean
