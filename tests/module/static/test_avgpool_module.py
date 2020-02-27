import nnabla as nn
from nnabla_nas.module import static as smo


def test_avgpool_module():
    shape = (10, 3, 32, 32)
    input = smo.Input(nn.Variable(shape))

    inp_module = smo.Input(value=input)
    pool = smo.AvgPool(parents=[inp_module],
                       kernel=(3, 3),
                       stride=(1, 1))
    assert pool.shape == (10, 3, 30, 30)


if __name__ == '__main__':
    test_avgpool_module()
