
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
import nnabla.functions as F
import nnabla.parametric_functions as PF

from nnabla_nas.utils.estimator.flops import FLOPsEstimator


def test_FLOPsEstimator():
    x = nn.Variable((1, 3, 12, 12))
    y = PF.depthwise_convolution(x, kernel=(5, 5), with_bias=True)
    t = PF.fused_batch_normalization(y)
    z = F.relu6(F.sigmoid(PF.affine(t, (3, 3), base_axis=2) + 3))
    z = F.global_average_pooling(z)

    est = FLOPsEstimator()
    assert est.predict(z) == 17644
