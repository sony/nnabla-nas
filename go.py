import nnabla as nn
import numpy as np

from nnabla_nas.contrib.mbn.modules import *
from nnabla_nas.contrib.mbn.network import SearchNet
from nnabla_nas.contrib.mbn.network import TrainNet
from nnabla_nas.utils import load_parameters

net = TrainNet(num_classes=10, mode='sample', genotype='log/mbn/search/arch.h5')

# print(net)

params = load_parameters('log/mbn/search/arch.h5')

for k, v in params.items():
    print(k, np.argmax(v.d.flatten()))
