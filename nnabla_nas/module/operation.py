from .module import Module


class Lambda(Module):
    r"""Lambda module.

    This module wraps a NNabla operator.

    Args:
        func (nnabla.functions): A NNabla funcion.
    """

    def __init__(self, func):
        self._func = func

    def call(self, *args, **kargs):
        return self._func(*args, **kargs)

    def extra_repr(self):
        return f'function={self._func.__name__}'
