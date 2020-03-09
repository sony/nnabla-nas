import nnabla as nn
from nnabla_nas.module import static as smo
from nnabla_nas.contrib import zoph


def test_sepconv3x3_module():
    shape = (10, 3, 32, 32)
    input = smo.Input(nn.Variable(shape))

    conv = zoph.SepConv3x3(parents=[input],
                           channels=64)

    assert conv.shape == (10, 64, 32, 32)


if __name__ == '__main__':
    test_sepconv3x3_module()