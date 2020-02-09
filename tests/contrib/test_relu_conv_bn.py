import nnabla as nn
import numpy as np
import pytest

from nnabla_nas.contrib.darts.modules import ReLUConvBN


@pytest.mark.parametrize('in_channels', [3, 5, 10])
@pytest.mark.parametrize('out_channels', [8, 16, 32])
def test_ReLUConvBN(in_channels, out_channels):
    module = ReLUConvBN(in_channels, out_channels, kernel=(3, 3), pad=(1, 1))
    input = nn.Variable((8, in_channels, 32, 32))
    output = module(input)

    assert output.shape == (8, out_channels, 32, 32)

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()
