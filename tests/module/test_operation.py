import nnabla as nn
import numpy as np
import nnabla.functions as F
from nnabla_nas.module import Lambda
import pytest


@pytest.mark.parametrize('func', [F.add2, F.sub2])
def test_Lambda(func):
    module = Lambda(func)
    input1 = nn.Variable((8, 3, 32, 32))
    input2 = nn.Variable((8, 3, 32, 32))

    output = module(input1, input2)

    assert isinstance(output, nn.Variable)

    input1.d = np.random.randn(*input1.shape)
    input2.d = np.random.randn(*input2.shape)

    print(module)

    output.forward()
    assert not np.isnan(output.d).any()
