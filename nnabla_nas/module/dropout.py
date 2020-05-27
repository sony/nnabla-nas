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


class Dropout(Module):
    r"""Dropout layer.

    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every
    forward call.

    Args:
        drop_prob (:obj:`int`, optional): The probability of an element to be
            zeroed. Defaults to 0.5.
    """

    def __init__(self, drop_prob=0.5):
        self._drop_prob = drop_prob

    def call(self, input):
        if self._drop_prob == 0 or not self.training:
            return input
        return F.dropout(input, self._drop_prob)

    def extra_repr(self):
        return f'drop_prob={self._drop_prob}'
