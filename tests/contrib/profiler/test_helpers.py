import pytest
import json

import nnabla as nn


@pytest.mark.parametrize("search_net_config", ["examples/mobilenet_cifar10_search.json"])
@pytest.mark.parametrize("mode", ["full", "sample"])
def test_get_search_net_on_config_file(search_net_config, mode):
    """Execution test
    """
    from nnabla_nas.contrib.profiler.helpers import get_search_net

    # SearchNet
    with open(search_net_config) as fp:
        config = json.load(fp)
    net_config = config['network'].copy()
    net = get_search_net(net_config, mode)
    inp = nn.Variable([1] + config["input_shape"])
    out = net(inp)
    assert isinstance(out, nn.Variable)
