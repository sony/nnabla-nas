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


class ReLU(Module):
    r"""ReLU layer.
    Applies the rectified linear unit function element-wise.

    Args:
        inplace (bool, optional): can optionally do the operation in-place.
            Default: ``False``.
    """

    def __init__(self, inplace=False):
        Module.__init__(self)
        self._inplace = inplace

    def call(self, input):
        return F.relu(input, inplace=self._inplace)

    def extra_repr(self):
        return f'inplace={self._inplace}'


class ReLU6(Module):
    r"""ReLU6 layer.
    Capping ReLU activation to 6 is often observed to learn sparse features
    earlier.

    """

    def __init__(self):
        super().__init__()

    def call(self, input):
        return F.relu6(input)


class LeakyReLU(Module):
    r"""LeakyReLU layer.
    Element-wise Leaky Rectified Linear Unit (ReLU) function.

    Args:
        alpha(float, optional): The slope value multiplied to negative numbers.
            :math:`\alpha` in the definition. Defaults to 0.1.
        inplace (bool, optional): can optionally do the operation in-place.
            Default: ``False``.
    """

    def __init__(self, alpha=0.1, inplace=False):
        self._alpha = alpha
        self._inplace = inplace

    def call(self, input):
        return F.leaky_relu(input, alpha=self._alpha, inplace=self._inplace)

    def extra_repr(self):
        return f'alpha={self._alpha}, inplace={self._inplace}'
