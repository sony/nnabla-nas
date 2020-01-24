import nnabla as nn
import pytest

from nnabla_nas.module import BatchNormalization, Parameter


@pytest.mark.parametrize('fix_parameters', [True, False])
def test_batchnorm(fix_parameters):
    module = BatchNormalization(
        n_features=5, n_dims=4, fix_parameters=fix_parameters)
    input = nn.Variable((8, 5, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape

    objcls = nn.Variable if fix_parameters else Parameter
    assert isinstance(module._beta, objcls)
    assert isinstance(module._gamma, objcls)

    assert isinstance(module._mean, nn.Variable)
    assert isinstance(module._var, nn.Variable)
