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

from nnabla_nas.contrib.classification.mobilenet import SearchNet


def test_mobilenet():
    net = SearchNet(num_classes=1000)
    input = nn.Variable((1, 3, 224, 224))

    assert net(input).shape == (1, net._num_classes)
    assert str(net)
