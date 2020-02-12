import nnabla as nn

from nnabla_nas.contrib.mbn.modules import *
from nnabla_nas.contrib.mbn.network import SearchNet

#m = ConvBNReLU(3, 5, (2, 2))
#m = Conv1x1BN(3, 5)
#m = InvertedResidual(3, 3, 1, 1)
m = SearchNet(num_classes=10)
x = nn.Variable((1, 3, 32, 32))

print(m(x).shape)
# print(m)
