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

from nnabla_nas.module import static as smo


def test_merging_module():
    shape = (10, 3, 32, 32)
    input_1 = smo.Input(nn.Variable(shape))
    input_2 = smo.Input(nn.Variable(shape))
    merge_add = smo.Merging([input_1, input_2],
                            mode='add')
    merge_con = smo.Merging([input_1, input_2],
                            mode='concat')
    assert merge_add().shape == (10, 3, 32, 32)
    assert merge_con().shape == (10, 6, 32, 32)


if __name__ == '__main__':
    test_merging_module()
