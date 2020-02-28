import nnabla as nn
from nnabla_nas.module import static as smo
from nnabla_nas.contrib import zoph
from nnabla_nas.contrib.zoph import ZOPH_CANDIDATES


def test_zophcell_module():
    shape = (10, 3, 32, 32)
    input_1 = smo.Input(nn.Variable(shape))
    input_2 = smo.Input(nn.Variable(shape))

    zc = zoph.ZophCell(parents=[input_1,
                                input_2],
                       candidates=ZOPH_CANDIDATES,
                       channels=64)
    assert zc().shape == (10, 320, 32, 32)
    assert zc.shape == (10, 320, 32, 32)


if __name__ == '__main__':
    test_zophcell_module()
