import nnabla as nn
import numpy as np
import pytest

from nnabla_nas.module import Merging


@pytest.mark.parametrize('mode', ['concat', 'add'])
def test_merging(mode):
    module = Merging(mode=mode, axis=int(mode == 'concat'))

    input_1 = nn.Variable((8, 5, 3))
    input_2 = nn.Variable((8, 5, 3))

    output = module(input_1, input_2)

    assert isinstance(output, nn.Variable)
    if mode == 'concat':
        assert output.shape == (8, 10, 3)
    else:
        assert output.shape == (8, 5, 3)

    input_1.d = np.random.randn(*input_1.shape)
    input_2.d = np.random.randn(*input_2.shape)
    output.forward()
    assert not np.isnan(output.d).any()
