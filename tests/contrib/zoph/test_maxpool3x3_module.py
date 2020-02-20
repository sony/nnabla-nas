import nnabla as nn
from nnabla_nas.module import static as smo
from nnabla_nas.contrib import zoph


def test_maxpool3x3_module():
    shape = (10, 3, 32, 32)
    input = smo.Input(nn.Variable(shape))

    pool = zoph.MaxPool3x3(parents=[input])

    assert pool.shape == (10, 3, 32, 32)


if __name__ == '__main__':
    test_maxpool3x3_module()
