import nnabla as nn
from nnabla_nas.contrib import zoph


def test_zoph_searchnet():
    shape = (10, 3, 32, 32)
    input = nn.Variable(shape)

    zn = zoph.SearchNet()
    assert zn.shape == (1, 10)
    out = zn(input._value)
    assert out.shape == (10, 10)


if __name__ == '__main__':
    test_zoph_searchnet()
