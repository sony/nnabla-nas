import nnabla as nn
from nnabla_nas.module import static as smo
from nnabla_nas.contrib import zoph


def test_sepconv_module():
    shape = (10, 3, 32, 32)
    input = smo.Input(nn.Variable(shape))

    conv = zoph.SepConv(parents=[input],
                        in_channels=3,
                        out_channels=64,
                        kernel=(3, 3))

    assert conv.shape == (10, 64, 30, 30)


if __name__ == '__main__':
    test_sepconv_module()
