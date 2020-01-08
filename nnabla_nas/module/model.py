from .module import Module


class Model(Module):
    def __init__(self):
        super().__init__()
        self._input_shape = None
        self._output_shape = None

    def get_net_paramters(self, grad_only=False):
        raise NotImplementedError

    def get_arch_parameters(self, grad_only=False):
        raise NotImplementedError

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape
