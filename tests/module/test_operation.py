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
import nnabla.functions as F
import numpy as np
import pytest

from nnabla_nas.module import Lambda


@pytest.mark.parametrize('func', [F.add2, F.sub2, F.mul2, F.div2])
def test_Lambda(func):
    module = Lambda(func)
    input1 = nn.Variable((8, 3, 32, 32))
    input2 = nn.Variable((8, 3, 32, 32))

    output = module(input1, input2)

    assert isinstance(output, nn.Variable)

    input1.d = np.random.randn(*input1.shape)
    input2.d = np.random.randn(*input2.shape)

    output.forward()
    assert not np.isnan(output.d).any()
