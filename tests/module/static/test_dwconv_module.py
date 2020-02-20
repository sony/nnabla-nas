import nnabla as nn
from nnabla_nas.module import static as smo


def test_dwconv_module():
    shape = (10, 3, 32, 32)
    input = smo.Input(nn.Variable(shape))

    inp_module = smo.Input(value=input)
    conv = smo.DwConv(parents=[inp_module],
                      in_channels=3,
                      kernel=(3, 3))

    assert conv.shape == (10, 3, 30, 30)


if __name__ == '__main__':
    test_dwconv_module()
