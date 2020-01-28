import nnabla as nn
import numpy as np

from nnabla_nas.module import AvgPool, GlobalAvgPool, MaxPool


def test_MaxPool():
    module = MaxPool(kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()


def test_AvgPool():
    module = AvgPool(kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()


def test_GlobalAvgPool():
    module = GlobalAvgPool()
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape[:2] + (1, 1)

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()
