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

import nnabla_nas.module as Mo
from nnabla_nas.utils.tensorboard import SummaryWriter


class MyModel(Mo.Module):
    def __init__(self):
        self.conv = Mo.Conv(3, 5, (3, 3), (1, 1))
        self.bn = Mo.BatchNormalization(5, 4)
        self.classifier = Mo.Sequential(
            Mo.ReLU(),
            Mo.GlobalAvgPool(),
            Mo.Linear(5, 10)
        )

    def call(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.classifier(out)
        return out


def test_graph():
    model = MyModel()
    inputs = nn.Variable([1, 3, 32, 32])
    writer = SummaryWriter('__nnabla_nas__/tensorboard')
    writer.add_graph(model, inputs)
    writer.add_scalar('accuracy', 1.0)

    writer.close()
