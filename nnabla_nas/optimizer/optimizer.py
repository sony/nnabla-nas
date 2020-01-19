from collections import OrderedDict

import nnabla.solvers as S


class Optimizer(object):

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
        self._solver.zero_grad()

    def get_parameters(self):
        return self._solver.get_parameters()

    def get_learning_rate(self):
        return self._lr_scheduler.get_learning_rate(self._iter)

    def clear_parameters(self):
        self._solver.clear_parameters()
        self._iter = 0
        if self._retain_state:
            self._states.clear()
