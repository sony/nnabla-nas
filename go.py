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

    def __call__(self, input):
        return self.weights - input


class Block(Module):
    def __init__(self, shape=(3, 3)):
        self.unit0 = BasicUnit(shape=shape)
        self.unit1 = BasicUnit(shape=shape)
        self.unit2 = BasicUnit(shape=shape)

    def __call__(self, input):
        out = self.unit0(input)
        out = self.unit1(out)
        out = self.unit2(out)
        return out + input


class MyModule(Module):
    def __init__(self, shape=(3, 3)):
        self.weights = Parameter(shape, initializer=np.random.randn(*shape))
        self.module1 = Block()
        self.module2 = Block()
        self.const = nn.Variable(shape, need_grad=False)
        self.shape = shape

    def __call__(self, input):
        out = self.module1(input)
        out = self.module2(out)
        return self.weights * out + self.const


m = MyModule()
m.need_grad = False
m.training = False
# m.apply(need_grad=False)

print(m.training)
print(m.__dict__)
