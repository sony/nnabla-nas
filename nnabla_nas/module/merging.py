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


class Merging(Module):
    r"""Merging layer.

    Merges a list of NNabla Variables.

    Args:
        mode (str): The merging mode ('concat', 'add', 'mul'), where `concat` indicates that the
            inputs will be concatenated, `add` means the element-wise addition, and `mul` means
            the element-wise multiplication.
        axis (int, optional): The axis for merging when 'concat' is used.
            Defaults to 1.
    """

    def __init__(self, mode, axis=1):
        if mode not in ('concat', 'add', 'mul'):
            raise KeyError(f'{mode} is not supported.')
        self._mode = mode
        self._axis = axis

    def call(self, *input):
        if self._mode == 'concat' and len(input) > 1:
            return F.concatenate(*input, axis=self._axis)
        if self._mode == 'add':
            return sum(input)
        out = input[0]
        for i in range(1, len(input)):
            out = F.mul2(out, input[i])
        return out

    def extra_repr(self):
        return f'mode={self._mode}, axis={self._axis}'
