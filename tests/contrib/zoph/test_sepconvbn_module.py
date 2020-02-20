import nnabla as nn
from nnabla_nas.module import static as smo
from nnabla_nas.contrib import zoph


def test_sepconvbn_module():
    shape = (10, 3, 32, 32)
    input = smo.Input(nn.Variable(shape))

    conv = zoph.SepConvBN(parents=[input],
                          out_channels=64,
                          kernel=(3, 3),
                          dilation=(1, 1))

    assert conv.shape == (10, 64, 32, 32)
    mod_names = [mi.name for _, mi in conv.get_modules() if
                 isinstance(mi, smo.Module)]
    ex_mod_names = ['',
                    '/SepConv_1',
                    '/SepConv_2',
                    '/bn',
                    '/relu']
    for mni, emni in zip(mod_names, ex_mod_names):
        assert mni == emni


if __name__ == '__main__':
    test_sepconvbn_module()
