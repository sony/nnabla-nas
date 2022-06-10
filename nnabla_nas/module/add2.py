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

import nnabla.functions as F
from .module import Module


class Add2(Module):
    r"""Adds two modules
    Args:
        inplace(bool): The output array is shared with the 1st input array
        name (string): the name of this module
    """

    def __init__(self, inplace=False, name=''):
        Module.__init__(self, name=name)
        self._scope_name = f'<add2 at {hex(id(self))}>'
        self._inplace = inplace

    def call(self, input1, input2):
        return F.add2(input1, input2, inplace=self._inplace)
