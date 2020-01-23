import nnabla as nn

from nnabla_nas.module import Linear


def test_linear():
    module = Linear(in_features=5, out_features=3)
    x = nn.Variable((8, 5))
    y = module(x)
    assert isinstance(y, nn.Variable)
    assert y.shape == (8, 3)
