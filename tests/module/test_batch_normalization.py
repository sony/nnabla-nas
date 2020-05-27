# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nnabla as nn
import numpy as np
import pytest

from nnabla_nas.module import BatchNormalization
from nnabla_nas.module import Parameter


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

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()
