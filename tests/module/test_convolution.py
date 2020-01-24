import nnabla as nn
import numpy as np
import pytest

from nnabla_nas.module import Conv, DwConv, Parameter


@pytest.mark.parametrize('fix_parameters', [True, False])
def test_convolution(fix_parameters):
    module = Conv(in_channels=3, out_channels=3, kernel=(3, 3),
                  pad=(1, 1), stride=(1, 1), fix_parameters=fix_parameters)
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape

    objcls = nn.Variable if fix_parameters else Parameter
    assert isinstance(module._W, objcls)
    assert isinstance(module._b, objcls)

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()


@pytest.mark.parametrize('fix_parameters', [True, False])
@pytest.mark.parametrize('base_axis', [1])
def test_depthwise_convolution(fix_parameters, base_axis):
    module = DwConv(in_channels=3, kernel=(3, 3), base_axis=base_axis,
                    pad=(2, 2), stride=(1, 1), fix_parameters=fix_parameters)
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape[base_axis] == input.shape[base_axis]

    objcls = nn.Variable if fix_parameters else Parameter
    assert isinstance(module._W, objcls)
    assert isinstance(module._b, objcls)

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()
