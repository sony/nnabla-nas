import nnabla as nn
from nnabla_nas.contrib.mobilenet import SearchNet


def test_mobilenet():
    net = SearchNet(num_classes=1000)
    input = nn.Variable((1, 3, 224, 224))

    assert net(input).shape == (1, net._num_classes)
    assert str(net)
