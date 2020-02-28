import nnabla as nn
from nnabla_nas.module import static as smo
from nnabla_nas.contrib import zoph


def test_avgpool3x3_module():
    shape = (10, 3, 32, 32)
    input = smo.Input(nn.Variable(shape))

    pool = zoph.AveragePool3x3(parents=[input])

    assert pool.shape == (10, 3, 32, 32)


if __name__ == '__main__':
    test_avgpool3x3_module()
