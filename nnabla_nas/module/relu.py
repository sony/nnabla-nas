import nnabla.functions as F

from .module import Module


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self._inplace = inplace

    def __call__(self, input):
        return F.relu(input, inplace=self._inplace)
