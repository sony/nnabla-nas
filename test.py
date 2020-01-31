import nnabla_nas.utils as ut
import numpy as np
import time
import nnabla as nn
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context
from nnabla import logger
from nnabla.initializer import ConstantInitializer
from scipy.special import softmax

from nnabla_nas.module import Module
from nnabla_nas.contrib.pnas.modules import SampledOp, CANDIDATE_FUNC

ctx = get_extension_context('cudnn', device_id=2)
nn.set_default_context(ctx)

from nnabla_nas.contrib.darts import SearchNet

model = SearchNet(3, 16, 4, 10)
x = nn.Variable([32, 3, 32, 32])

start = time.time()
for i in range(100):
    y = model(x)
    #y.forward()
end = time.time()

print(end-start)