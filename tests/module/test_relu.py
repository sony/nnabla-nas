import nnabla as nn

from nnabla_nas.module import LeakyReLU, ReLU, ReLU6


def test_ReLU():
    module = ReLU()
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape


def test_ReLU6():
    module = ReLU6()
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape


def test_LeakyReLU():
    module = LeakyReLU(alpha=0.3, inplace=True)
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape
