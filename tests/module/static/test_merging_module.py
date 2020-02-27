import nnabla as nn
from nnabla_nas.module import static as smo


def test_merging_module():
    shape = (10, 3, 32, 32)
    input_1 = smo.Input(nn.Variable(shape))
    input_2 = smo.Input(nn.Variable(shape))
    merge_add = smo.Merging([input_1, input_2],
                            mode='add')
    merge_con = smo.Merging([input_1, input_2],
                            mode='concat')
    assert merge_add().shape == (10, 3, 32, 32)
    assert merge_con().shape == (10, 6, 32, 32)


if __name__ == '__main__':
    test_merging_module()
