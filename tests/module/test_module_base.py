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

from collections import OrderedDict
import os

import nnabla as nn
from nnabla.testing import assert_allclose
import numpy as np
import pytest

from nnabla_nas.module import Parameter
from nnabla_nas.module.module import Module


class BasicUnit(Module):
    def __init__(self, shape=(3, 3)):
        self.weights = Parameter(shape, initializer=np.random.randn(*shape))
        self.shape = shape

    def call(self, input):
        return self.weights + input


class Block(Module):
    def __init__(self, shape=(3, 3)):
        self.unit0 = BasicUnit(shape=shape)
        self.unit1 = BasicUnit(shape=shape)
        self.unit2 = BasicUnit(shape=shape)

    def call(self, input):
        out = self.unit0(input)
        out = self.unit1(out)
        out = self.unit2(out)
        return out


class MyModule(Module):
    def __init__(self, shape=(3, 3)):
        self.weights = Parameter(shape, initializer=np.random.randn(*shape))
        self.module1 = Block(shape)
        self.module2 = Block(shape)
        self.const = nn.Variable(shape, need_grad=False)
        self.shape = shape

    def call(self, input):
        out = self.module1(input)
        out = self.module2(out)
        out = out + self.weights + self.const
        return out


def test_module():
    # test save and load functions
    module = MyModule(shape=(5, 5))
    assert isinstance(module.get_parameters(), OrderedDict)
    assert len(module.get_parameters()) == 7
    assert len([_ for _ in module.get_modules()]) == 9


def test_load_save_parameters():
    module = MyModule(shape=(5, 5))
    params = module.get_parameters()

    if not os.path.exists('__nnabla_nas__'):
        os.makedirs('__nnabla_nas__')
    nn.save_parameters('__nnabla_nas__/params.h5', params)
    nn.load_parameters('__nnabla_nas__/params.h5')

    params0 = nn.get_parameters()
    for k, v in params.items():
        assert_allclose(v.d, params0[k].d)


def test_set_parameters():
    # test set_parameters function
    module = MyModule(shape=(5, 5))
    params0 = module.get_parameters()
    for k, v in params0.items():
        v.d = np.random.randn(*v.shape)
    module.set_parameters(params0)
    for k, v in module.get_parameters().items():
        assert_allclose(v.d, params0[k].d)


@pytest.mark.parametrize('prop', ['training', 'need_grad'])
@pytest.mark.parametrize('mode', [True, False])
def test_properties(prop, mode):
    module = MyModule(shape=(5, 5))
    module.apply(**{prop: mode})
    for k, m in module.get_modules():
        assert getattr(m, prop) == mode


def test_inputs():
    input_shape = (5, 5)
    module = MyModule(input_shape)
    inputs = nn.Variable(input_shape)
    module(inputs)

    assert input_shape == module.input_shapes[0]
