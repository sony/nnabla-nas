import torch
from torch import nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, 1)
        self.feat = nn.Linear(900, 1)

    def forward(self, x):
        x = self.conv(x)
        return self.feat(x.view(3, -1))


net = MyModel()
x = torch.tensor(np.random.randn(1, 3, 32, 32)).float()

s = SummaryWriter('torch')
s.add_graph(net, x)

s.close()
