import nnabla as nn
import numpy as np
import pytest

from nnabla_nas.contrib.misc import AuxiliaryHeadCIFAR


@pytest.mark.parametrize('in_channels', [8, 16, 32])
@pytest.mark.parametrize('num_classes', [2, 10, 100])
def test_AuxiliaryHeadCIFAR(in_channels, num_classes):
    module = AuxiliaryHeadCIFAR(in_channels, num_classes=num_classes)
    input = nn.Variable((16, in_channels, 8, 8))
    output = module(input)

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()
    assert output.shape == (16, num_classes)
