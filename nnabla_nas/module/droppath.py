import nnabla.functions as F

from .module import Module


class DropPath(Module):
    def __init__(self, drop_prob, batch_size):
        super().__init__()
        self._batch_size = batch_size
        self.keep_prob = 1.0 - drop_prob
        rand = F.rand(shape=(batch_size, 1, 1, 1))
        self.mask = F.greater_equal_scalar(rand, drop_prob)

    def __call__(self, input):
        out = F.mul_scalar(input, 1./self.keep_prob)
        out = F.mul2(out, self.mask)
        return out
