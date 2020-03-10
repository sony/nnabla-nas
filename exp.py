import nnabla as nn
from nnabla_nas import module as Mo


class Net(Mo.Module):

    def __init__(self):
        self.fc = Mo.Linear(10, 5)

    self.parameters = Mo.ParameterList([
        Mo.Parameter((1, 2)),
        Mo.Parameter((1, 2))
    ])

    self.modules = Mo.ModuleList([
        Mo.Conv(3, 3, (3, 3)),
        Mo.Conv(3, 5, (3, 3)),
    ])

    def call(self, input):
        return self.fc(input) * self.coef


x = nn.Variable((1, 10))
net = Net()

print(net)
print(net(x))
