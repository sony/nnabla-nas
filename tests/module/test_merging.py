import nnabla as nn
import pytest

from nnabla_nas.module import Merging


@pytest.mark.parametrize('mode', ['concat', 'add'])
def test_merging(mode):
    module = Merging(mode=mode, axis=int(mode == 'concat'))

    x = nn.Variable((8, 5, 3))
    y = nn.Variable((8, 5, 3))

    out = module(x, y)

    assert isinstance(out, nn.Variable)
    if mode == 'concat':
        assert out.shape == (8, 10, 3)
    else:
        assert out.shape == (8, 5, 3)
