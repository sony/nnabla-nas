import nnabla as nn
from nnabla_nas.module import static as smo


def test_batchnorm_module():
    input = smo.Input(value=nn.Variable((10, 3, 32, 32)))
    bn = smo.BatchNormalization(parents=[input],
                                n_features=3,
                                n_dims=4)
    output = bn()

    assert output.shape == (10, 3, 32, 32)


if __name__ == '__main__':
    test_batchnorm_module()
