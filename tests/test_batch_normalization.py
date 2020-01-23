import nnabla as nn

from nnabla_nas.module import BatchNormalization


def test_batchnorm():
    op = BatchNormalization(n_features=5, n_dims=4, fix_parameters=False)
    x = nn.Variable((8, 5, 32, 32))
    y = op(x)
    assert isinstance(y, nn.Variable)
    assert y.shape == x.shape
