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
