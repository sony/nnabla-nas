from collections import OrderedDict
import time

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla import logger
from nnabla.ext_utils import get_extension_context
from nnabla.initializer import ConstantInitializer
from scipy.special import softmax

import nnabla_nas.utils as ut

from nnabla_nas.contrib.pnas.modules import CANDIDATE_FUNC, SampledOp
from nnabla_nas.module import Module, Conv, Zero, Identity, Linear
import json

from nnabla_nas.runner.search_pnas import Searcher

ctx = get_extension_context('cudnn', device_id=1)
nn.set_default_context(ctx)


class Model(Module):
    def __init__(self):
        super().__init__()
        self.input = SampledOp(operators=[
            Conv(3, 3, (3, 3), (1, 1)),
            Zero(),
            # Identity()
        ])
        self.linear = Linear(3*32*32, 10)

    def get_net_parameters(self, grad_only=False):
        param = OrderedDict()
        for key, val in self.get_parameters(grad_only).items():
            if '_alpha' not in key:
                param[key] = val
        return param

    def get_arch_parameters(self, grad_only=False):
        param = OrderedDict()
        for key, val in self.get_parameters(grad_only).items():
            if '_alpha' in key:
                param[key] = val
        return param

    def call(self, input):
        out = self.input(input)
        out = self.linear(out)
        return out


model = Model()
conf = json.load(open('examples/tests.json'))

Searcher(model, conf).run()

nn.load_parameters('log/pnas-con/search/arch.h5')
x = nn.get_parameters()

print(x['input/_alpha'].d)
