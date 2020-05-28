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

from nnabla_nas.module import Module
from nnabla_nas.module import ModuleList
from nnabla_nas.module import Parameter
from nnabla_nas.module import ParameterList
from nnabla_nas.module import Sequential


class BasicUnit(Module):
    def __init__(self, shape=(3, 3)):
        self.weights = Parameter(shape, initializer=np.random.randn(*shape))
        self.shape = shape

    def call(self, input):
        return self.weights + input

    def __extra_repr__(self):
        return f'shape={self.shape}'


class Block(Module):
    def __init__(self, shape=(3, 3)):
        self.blocks = ModuleList([
            BasicUnit(shape=shape),
            BasicUnit(shape=shape),
            BasicUnit(shape=shape)
        ])

    def call(self, input):
        out = input
        for module in self.blocks:
            out = module(out)
        return out


class BlockParameter(Module):
    def __init__(self, shape=(3, 3)):
        self.blocks = ParameterList([
            Parameter(shape),
            Parameter(shape),
            Parameter(shape)
        ])

    def call(self, input):
        return input + sum(self.blocks)


def test_ModuleList():

    class MyModule(Module):
        def __init__(self, shape=(3, 3)):
            self.weights = Parameter(
                shape, initializer=np.random.randn(*shape))
            self.module1 = Block(shape)
            self.module2 = Block(shape)
            self.const = nn.Variable(shape, need_grad=False)
            self.shape = shape

        def call(self, input):
            out = self.module1(input)
            out = self.module2(out)
            out = out + self.weights + self.const
            return out

    shape = (3, 3)
    module = MyModule(shape)
    block = module.module1.blocks

    # test the len function
    assert len([_ for _ in module.get_modules()]) == 11
    assert len(block) == 3

    # test the append function
    block.append(Block(shape))
    assert len(block) == 4
    assert len([_ for _ in module.get_modules()]) == 16

    # test the extend function
    block += [Block(shape), Block(shape)]
    assert len(block) == 6
    assert len([_ for _ in module.get_modules()]) == 26

    # test the insert function
    b = BasicUnit(shape)
    block.insert(2, b)
    assert block[2] is b

    # test the assign function
    b = Block(shape)
    block[1] = b
    assert block[1] is b

    # test the delete function
    del block[1]
    assert len(block) == 6
    # test the module
    x = nn.Variable(shape)
    y = module(x)
    assert isinstance(y, nn.Variable)


def test_ParameterList():

    class MyModule(Module):
        def __init__(self, shape=(3, 3)):
            self.mo = BlockParameter(shape)
            self.const = nn.Variable(shape, need_grad=False)
            self.shape = shape

        def call(self, input):
            out = self.mo(input)
            out = out + self.const
            return out

    shape = (3, 3)
    module = MyModule(shape)
    block = module.mo.blocks

    # test the len function
    assert len([_ for _ in module.get_modules()]) == 3
    assert len(block) == 3

    # test the append function
    block.append(Parameter(shape))
    assert len(block) == 4
    assert len([_ for _ in module.get_modules()]) == 3

    # test the extend function
    block += [Parameter(shape), Parameter(shape)]
    assert len(block) == 6
    assert len([_ for _ in module.get_modules()]) == 3

    # test the insert function
    b = Parameter(shape)
    block.insert(2, b)
    assert block[2] is b

    # test the assign function
    b = Block(shape)
    block[1] = b
    assert block[1] is b

    # test the delete function
    del block[1]
    assert len(block) == 6
    # test the module
    x = nn.Variable(shape)
    y = module(x)
    assert isinstance(y, nn.Variable)


def test_Sequential():
    class MyModel(Module):
        def __init__(self, shape):
            self.feat = Sequential(
                BasicUnit(shape),
                Block(shape),
                BasicUnit(shape)
            )

        def call(self, input):
            return self.feat(input)

    shape = (5, 5)
    model = MyModel(shape)
    x = nn.Variable(shape)

    # test the delete function
    del model.feat[1]
    assert len(model.feat) == 2
    # test append
    model.feat.append(Block(shape))
    assert len(model.feat) == 3

    # test insert
    model.feat.insert(0, Block(shape))
    assert len(model.feat) == 4

    y = model(x)
    assert isinstance(y, nn.Variable)
