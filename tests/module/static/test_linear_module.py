import nnabla as nn
from nnabla_nas.module import static as smo


def test_linear_module():
    input = smo.Input(value=nn.Variable((8, 5)))
    linear = smo.Linear(parents=[input],
                        in_features=5,
                        out_features=3)
    output = linear()

    assert output.shape == (8, 3)


if __name__ == '__main__':
    test_linear_module()
