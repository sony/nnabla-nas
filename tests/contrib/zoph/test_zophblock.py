import nnabla as nn
from nnabla_nas.module import static as smo
from nnabla_nas.contrib import zoph
from nnabla_nas.contrib.zoph import ZOPH_CANDIDATES


def test_zophblock_module():
    shape = (10, 3, 32, 32)
    input_1 = smo.Input(nn.Variable(shape))
    input_2 = smo.Input(nn.Variable(shape))

    zb = zoph.ZophBlock(parents=[input_1,
                                 input_2],
                        candidates=ZOPH_CANDIDATES,
                        channels=64)

    out = zb()
    assert out.shape == (10, 64, 32, 32)
    assert zb.shape == (10, 64, 32, 32)
    assert len(zb) == 13
    cand = zb[4:-1]
    for ci, cri in zip(cand, ZOPH_CANDIDATES):
        assert type(ci) == cri


if __name__ == '__main__':
    test_zophblock_module()
