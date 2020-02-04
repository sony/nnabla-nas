import nnabla.functions as F

from .module import Module


class Dropout(Module):
    r"""Dropout layer.

    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every
    forward call.

    Args:
        drop_prob (:obj:`int`, optional): The probability of an element to be
            zeroed. Defaults to 0.5.
    """
    def __init__(self, drop_prob=0.5):
        self._drop_prob = drop_prob

    def call(self, input):
        if self._drop_prob == 0 or not self.training:
            return input
        return F.dropout(input, self._drop_prob)

    def extra_repr(self):
        return f'drop_prob={self._drop_prob}'
