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

from nnabla_nas.contrib import zoph
from nnabla_nas.contrib.classification.zoph import ZOPH_CANDIDATES
from nnabla_nas.module import static as smo


def test_zophblock_module():
    shape = (10, 3, 32, 32)
    input_1 = smo.Input(nn.Variable(shape))
    input_2 = smo.Input(nn.Variable(shape))

    zb = zoph.ZophBlock(parents=[input_1,
                                 input_2],
                        candidates=ZOPH_CANDIDATES,
                        channels=64)

    out = zb()
    assert out.shape == (10, 64, 32, 32)
    assert zb.shape == (10, 64, 32, 32)
    assert len(zb) == 13
    cand = zb[4:-1]
    for ci, cri in zip(cand, ZOPH_CANDIDATES):
        assert type(ci) == cri


if __name__ == '__main__':
    test_zophblock_module()
