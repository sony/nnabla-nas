import nnabla as nn
import numpy as np
import pytest

from nnabla_nas.module import Conv, Module, Dropout


class Block(Module):
    def __init__(self, in_channels, out_channels):
        self._conv = Conv(in_channels, out_channels, (3, 3), pad=(1, 1))
        self._drop = Dropout(drop_prob=0.3)

    def call(self, input):
        out = self._conv(input)
        out = self._drop(out)
        return out


@pytest.mark.parametrize('in_channels', [3, 5, 10])
@pytest.mark.parametrize('out_channels', [8, 16, 32])
def test_mixedop(in_channels, out_channels):
    module = Block(in_channels, out_channels)
    input = nn.Variable([8, in_channels, 32, 32])

    output = module(input)
    assert output.shape == (8, out_channels, 32, 32)

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()
