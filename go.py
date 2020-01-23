from collections import OrderedDict

import nnabla as nn
import numpy as np
import pytest
from nnabla.testing import assert_allclose

from nnabla_nas.module import Parameter
from nnabla_nas.module.module import Module


class BasicUnit(Module):
    def __init__(self, shape=(3, 3)):
        self.weights = Parameter(shape, initializer=np.random.randn(*shape))
        self.shape = shape

    def call(self, input):
        print(input.shape, self.weights.shape)
        return self.weights + input

    def __extra_repr__(self):
        return 'shape=' + str(self.shape)


class Block(Module):
    def __init__(self, shape=(3, 3)):
        self.unit0 = BasicUnit(shape=shape)
        self.unit1 = BasicUnit(shape=shape)
        self.unit2 = BasicUnit(shape=shape)
        self.shape = shape

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
        self.module3 = BasicUnit(shape)
        self.const = nn.Variable(shape, need_grad=False)
        self.shape = shape

    def call(self, input):
        out = self.module1(input)
        out = self.module2(out)
        out = out + self.weights + self.const
        return out


shape = (5, 5)
m = MyModule(shape)
print(m)
