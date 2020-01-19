from collections import OrderedDict

import nnabla.solvers as S


class Solver(object):
    """Manager of the solvers"""

    def __init__(self, optim, lr, **kargs):
        self._states = dict()
        self._solver = S.__dict__[optim](lr, **kargs)
        self._solver_init = S.__dict__[optim](lr, **kargs)

    def set_parameters(self, params, **kargs):
        params = self._clean_params(params)
        self._update_current_states()
        self._solver.set_parameters(
            params, reset=True, retain_state=False
        )
        if kargs.get('retain_state', False):
            states = self._get_current_states(params)
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

    def clip_grad_by_norm(self, v):
        return self._solver.clip_grad_by_norm(v)

    def _get_current_states(self, params):
        states = OrderedDict()
        self._solver_init.set_parameters(params)
        initial_states = self._solver_init.get_states()
        for k in params:
            states[k] = self._states.get(k, initial_states[k])
        return states

    def _update_current_states(self):
        self._states.update(self._solver.get_states())

    def _clean_params(self, params):
        clean, mark = OrderedDict(), set()
        for k, p in params.items():
            if p not in mark:
                mark.add(p)
                clean[k] = p
        return clean

    def clear_parameters(self):
        self._solver.clear_parameters()
        self._solver_init.clear_parameters()
        self._states.clear()
