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


def test_collapse_module():
    shape = (10, 3, 1, 1)
    input = smo.Input(nn.Variable(shape))

    inp_module = smo.Input(value=input)
    collapse = smo.Collapse(parents=[inp_module])

    assert collapse.shape == (10, 3)


if __name__ == '__main__':
    test_collapse_module()
