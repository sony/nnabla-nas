import nnabla as nn

from nnabla_nas.module import LeakyReLU, ReLU, ReLU6


def test_ReLU():
    module = ReLU()
    x = nn.Variable((8, 3, 32, 32))
    y = module(x)

    assert isinstance(y, nn.Variable)
    assert y.shape == x.shape


def test_ReLU6():
    module = ReLU6()
    x = nn.Variable((8, 3, 32, 32))
    y = module(x)

    assert isinstance(y, nn.Variable)
    assert y.shape == x.shape


def test_LeakyReLU():
    module = LeakyReLU(alpha=0.3, inplace=True)
    x = nn.Variable((8, 3, 32, 32))
    y = module(x)

    assert isinstance(y, nn.Variable)
    assert y.shape == x.shape
