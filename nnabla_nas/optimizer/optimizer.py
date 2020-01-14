import nnabla.functions as F


class Optimizer(object):

    def __init__(self, solver,
                 grad_clip=None,
                 weight_decay=None,
                 lr_scheduler=None):
        self.solver = solver
        self._grad_clip = grad_clip
        self._lr_scheduler = lr_scheduler
        self._weight_decay = weight_decay

    def set_parameters(self, params, **kargs):
        self.solver.set_parameters(params, **kargs)

    def get_parameters(self):
        return self.solver.get_parameters()

    def get_learning_rate(self, n_iter):
        return self._lr_scheduler.get_learning_rate(n_iter)

    def zero_grad(self):
        self.solver.zero_grad()

    def update(self, n_iter):
        if self._lr_scheduler is not None:
            lr = self.get_learning_rate(n_iter)
            self.solver.set_learning_rate(lr)

        if self._grad_clip is not None:
            self.solver.clip_grad_by_norm(self._grad_clip)

        if self._weight_decay is not None:
            self.solver.weight_decay(self._weight_decay)

        self.solver.update()
