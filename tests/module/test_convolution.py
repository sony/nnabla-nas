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

from nnabla_nas.module import Conv
from nnabla_nas.module import DwConv
from nnabla_nas.module import Parameter


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
