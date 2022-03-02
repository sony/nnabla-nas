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


class Hsigmoid(Module):
    r"""Hswish layer.
    Args:
        name (string): the name of this module
    """
    def __init__(self, name=''):
        self._name = name
        self._scope_name = f'<hsigmoid at {hex(id(self))}>'

    def call(self, x):
        return F.relu6(x + 3.) / 6


class Hswish(Module):
    r"""Hswish layer.
    Args:
        name (string): the name of this module
    """
    def __init__(self, name=''):
        self._name = name
        self._scope_name = f'<hswish at {hex(id(self))}>'

    def call(self, x):
        return x * F.relu6(x + 3.) / 6
