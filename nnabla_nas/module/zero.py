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


class Zero(Module):
    r"""Zero layer.
    A placeholder zero operator that is argument-insensitive.

    Args:
        stride (:obj:`tuple` of :obj:`int`, optional): Stride sizes for
            dimensions. Defaults to (1, 1).
    """

    def __init__(self, stride=(1, 1), *args, **kwargs):
        self._stride = stride

    def call(self, input):
        if self._stride[0] > 1:
            input = F.max_pooling(input, kernel=(1, 1), stride=self._stride)
        return F.mul_scalar(input, 0.0)

    def extra_repr(self):
        return f'stride={self._stride}'
