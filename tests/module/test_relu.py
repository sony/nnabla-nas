import nnabla as nn
import numpy as np

from nnabla_nas.module import LeakyReLU
from nnabla_nas.module import ReLU
from nnabla_nas.module import ReLU6


def test_ReLU():
    module = ReLU()
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()


def test_ReLU6():
    module = ReLU6()
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()


def test_LeakyReLU():
    module = LeakyReLU(alpha=0.3, inplace=True)
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()
