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

from nnabla_nas.module import LeakyReLU
from nnabla_nas.module import ReLU
from nnabla_nas.module import ReLU6


def test_ReLU():
    module = ReLU()
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()


def test_ReLU6():
    module = ReLU6()
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()


def test_LeakyReLU():
    module = LeakyReLU(alpha=0.3, inplace=True)
    input = nn.Variable((8, 3, 32, 32))
    output = module(input)

    assert isinstance(output, nn.Variable)
    assert output.shape == input.shape

    input.d = np.random.randn(*input.shape)
    output.forward()
    assert not np.isnan(output.d).any()
