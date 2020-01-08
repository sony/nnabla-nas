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

    def set_parameters(self, params):
        self.solver.set_parameters(params)

    def get_parameters(self):
        return self.solver.get_parameters()

    def zero_grad(self):
        self.solver.zero_grad()

    def update(self, n_iter):
        if self._lr_scheduler is not None:
            lr = self._lr_scheduler.get_learning_rate(n_iter)
            self.solver.set_learning_rate(lr)

        if self._grad_clip is not None:
            params = self.solver.get_parameters()
            for v in params.values():
                v.grad.copy_from(F.clip_by_norm(v.grad, self._grad_clip))

        if self._weight_decay is not None:
            self.solver.weight_decay(self._weight_decay)

        self.solver.update()
