import nnabla as nn

from nnabla_nas.module import Conv


def test_convolution():
    module = Conv(in_channels=3, out_channels=3,
                  kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    x = nn.Variable((8, 3, 32, 32))
    y = module(x)

    assert isinstance(y, nn.Variable)
    assert y.shape == x.shape
