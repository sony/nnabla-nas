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

import pytest
import json

import nnabla as nn


@pytest.mark.parametrize("search_net_config", ["examples/mobilenet_cifar10_search.json"])
@pytest.mark.parametrize("mode", ["full", "sample"])
def test_get_search_net_on_config_file(search_net_config, mode):
    """Execution test
    """
    from nnabla_nas.utils.profiler.helpers import get_search_net

    # SearchNet
    with open(search_net_config) as fp:
        config = json.load(fp)
    net_config = config['network'].copy()
    net = get_search_net(net_config, mode)
    inp = nn.Variable([1] + config["input_shape"])
    out = net(inp)
    assert isinstance(out, nn.Variable)
