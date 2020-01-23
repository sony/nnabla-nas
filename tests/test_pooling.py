import nnabla as nn

from nnabla_nas.module import AvgPool, GlobalAvgPool, MaxPool


def test_MaxPool():
    module = MaxPool(kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    x = nn.Variable((8, 3, 32, 32))
    y = module(x)

    assert isinstance(y, nn.Variable)
    assert y.shape == x.shape


def test_AvgPool():
    module = AvgPool(kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    x = nn.Variable((8, 3, 32, 32))
    y = module(x)

    assert isinstance(y, nn.Variable)
    assert y.shape == x.shape


def test_GlobalAvgPool():
    module = GlobalAvgPool()
    x = nn.Variable((8, 3, 32, 32))
    y = module(x)

    assert isinstance(y, nn.Variable)
    assert y.shape == x.shape[:2] + (1, 1)
