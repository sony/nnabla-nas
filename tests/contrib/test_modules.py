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
import nnabla.parametric_functions as PF
import pytest

from nnabla_nas import module as Mo
from nnabla_nas.contrib.classification.darts.modules import FactorizedReduce


@pytest.mark.parametrize('batch_size', [8, 16, 32])
@pytest.mark.parametrize('in_channels', [8, 16, 32])
@pytest.mark.parametrize('out_channels', [8, 16, 32])
@pytest.mark.parametrize('width', [8, 16, 32])
@pytest.mark.parametrize('height', [8, 16, 32])
def test_FactorizeReduce(batch_size, in_channels, out_channels, width, height):
    m = FactorizedReduce(in_channels=in_channels, out_channels=out_channels)
    x = nn.Variable([batch_size, in_channels, width, height])
    assert m(x).shape == (batch_size, out_channels, width // 2, height // 2)


@pytest.mark.parametrize('batch_size', [8, 16, 32])
@pytest.mark.parametrize('in_channels', [8, 16, 32])
@pytest.mark.parametrize('out_channels', [8, 16, 32])
@pytest.mark.parametrize('kernel', [(3, 3), (5, 5)])
@pytest.mark.parametrize('stride', [(1, 1), (2, 2)])
@pytest.mark.parametrize('pad', [(0, 0), (1, 1), (2, 2)])
def test_Convolution(batch_size, in_channels, out_channels, kernel, stride,
                     pad):
    nn.clear_parameters()
    x = nn.Variable([batch_size, in_channels, 64, 64])
    m = Mo.Conv(in_channels=in_channels, out_channels=out_channels,
                kernel=kernel, stride=stride, pad=pad)
    out = PF.convolution(x, out_channels, kernel=kernel,
                         stride=stride, pad=pad)
    assert m(x).shape == out.shape


@pytest.mark.parametrize('batch_size', [8, 16, 32])
@pytest.mark.parametrize('in_channels', [8, 16, 32])
@pytest.mark.parametrize('stride', [(1, 1), (2, 2)])
def test_Zero(batch_size, in_channels, stride):
    x = nn.Variable([batch_size, in_channels, 64, 64])
    m = Mo.Zero(stride=stride)
    if stride == (1, 1):
        assert m(x).shape == (batch_size, in_channels, 64, 64)
    else:
        assert m(x).shape == (batch_size, in_channels, 32, 32)
