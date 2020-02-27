import numpy as np
import nnabla as nn
from nnabla_nas.module import static as smo
from nnabla_nas.module.parameter import Parameter


def test_join_module():
    shape = (10, 3, 32, 32)
    input_1 = smo.Input(nn.Variable(shape))
    input_2 = smo.Input(nn.Variable(shape))
    alpha = Parameter(shape=(2,))
    alpha.d = np.array([1, 2])

    join_linear = smo.Join([input_1, input_2],
                           join_parameters=alpha,
                           mode='linear')
    join_sample = smo.Join([input_1, input_2],
                           join_parameters=alpha,
                           mode='sample')
    join_max = smo.Join([input_1, input_2],
                        join_parameters=alpha,
                        mode='max')

    assert join_linear().shape == (10, 3, 32, 32)
    assert join_sample().shape == (10, 3, 32, 32)
    assert join_max().shape == (10, 3, 32, 32)

    assert join_max._idx == 1
    assert join_max() == input_2._value


if __name__ == '__main__':
    test_join_module()
