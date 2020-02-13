import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.testing import assert_allclose

# from nnabla_nas.contrib.mbn.modules import *
# from nnabla_nas.contrib.mbn.network import SearchNet
# from nnabla_nas.contrib.mbn.network import TrainNet
# from nnabla_nas.utils import load_parameters

# net = TrainNet(num_classes=10, mode='sample', genotype='log/mbn/search/arch.h5')

# # print(net)

# params = load_parameters('log/mbn/search/arch.h5')

# for k, v in params.items():
#     print(k, np.argmax(v.d.flatten()))



x = np.arange(27).reshape(1, 3, 3, 3)
m = np.arange(3).reshape(1, 3, 1, 1)

x_var = nn.Variable.from_numpy_array(x)
m_var = nn.Variable.from_numpy_array(m)
out_var = F.sub2(x_var, m_var)

out_var.forward()

print(x)
print(out_var.d)

print(x - m)
