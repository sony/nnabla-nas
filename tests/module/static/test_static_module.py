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

from nnabla_nas import module as mo
from nnabla_nas.module import static as smo


def test_static_module():
    module_1 = smo.Module(name='module_1')
    module_2 = smo.Module(parents=[module_1],
                          name='module_2')

    assert module_1.children[0] == module_2
    assert module_2.parents[0] == module_1

    class MyModule(smo.Module):
        def __init__(self, parents):
            smo.Module.__init__(self, parents=parents)
            self.linear = mo.Linear(in_features=5, out_features=3)

        def call(self, *input):
            return self.linear(*input)

    input = smo.Input(value=nn.Variable((8, 5)))
    my_mod = MyModule(parents=[input])
    output = my_mod()

    assert 'linear' in my_mod.modules
    assert len(my_mod.modules) == 1
    assert output.shape == (8, 3)
    assert my_mod.shape == (8, 3)

    my_mod.reset_value()
    assert my_mod._value is None
    assert my_mod._shape == -1
    input.value = nn.Variable((10, 5))
    assert my_mod.shape == (10, 3)

    params = my_mod.get_parameters()
    assert len(params) == 2


if __name__ == '__main__':
    test_static_module()
