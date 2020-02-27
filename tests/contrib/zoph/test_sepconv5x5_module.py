import nnabla as nn
from nnabla_nas.module import static as smo
from nnabla_nas.contrib import zoph


def test_sepconv5x5_module():
    shape = (10, 3, 32, 32)
    input = smo.Input(nn.Variable(shape))

    conv = zoph.SepConv5x5(parents=[input],
                           channels=64)

    assert conv.shape == (10, 64, 32, 32)


if __name__ == '__main__':
    test_sepconv5x5_module()
