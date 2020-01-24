import nnabla as nn
import pytest

from nnabla_nas.module import Linear, Parameter


@pytest.mark.parametrize('fix_parameters', [True, False])
def test_linear(fix_parameters):
    module = Linear(in_features=5, out_features=3)
    input = nn.Variable((8, 5))
    output = module(input)
    assert isinstance(output, nn.Variable)
    assert output.shape == (8, 3)

    objcls = nn.Variable if fix_parameters else Parameter
    assert isinstance(module._W, objcls)
    assert isinstance(module._b, objcls)
