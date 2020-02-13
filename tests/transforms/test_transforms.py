from nnabla_nas.dataset.transforms import Normalize
import numpy as np
import nnabla as nn
from nnabla.testing import assert_allclose


def test_normalize():
    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    scale = 1./255

    norm = Normalize(mean=mean, std=std, scale=scale)

    input = np.random.randn(16, 3, 32, 32)
    m = np.reshape(mean, (1, 3, 1, 1))
    s = np.reshape(std, (1, 3, 1, 1))
    output = (input * scale - m) / s

    input_var = nn.Variable.from_numpy_array(input)
    output_var = norm(input_var)
    output_var.forward()

    assert_allclose(output_var.d, output)
