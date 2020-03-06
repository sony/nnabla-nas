import shutil
from collections import defaultdict
import nnabla as nn
from nnabla_nas.module.module import Module
from nnabla_nas.module import Parameter
import nnabla_nas.module as Mo
import numpy as np

from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.versions_pb2 import VersionDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.types_pb2 import DT_FLOAT
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto import event_pb2

from nnabla_nas.utils.tensorboard.writer import FileWriter
from nnabla_nas.utils.tensorboard.proto_graph import node_proto, tensor_shape_proto
from nnabla_nas.utils.tensorboard.nnabla_graph import GraphVisitor

# stepstats = RunMetadata(step_stats=StepStats(
#     dev_stats=[DeviceStepStats(device="/device:CPU:0")]))


class BasicUnit(Module):
    def __init__(self, shape=(3, 3)):
        self.weights = Parameter(shape, initializer=np.random.randn(*shape))
        self.shape = shape

    def call(self, input):
        return self.weights + input


class Block(Module):
    def __init__(self, shape=(3, 3)):
        self.unit0 = BasicUnit(shape=shape)
        self.unit1 = BasicUnit(shape=shape)
        self.unit2 = BasicUnit(shape=shape)

    def call(self, input):
        out = self.unit0(input)
        out = self.unit1(out)
        out = self.unit2(out)
        return out


class MyModule(Module):
    def __init__(self, shape=(3, 3)):
        self.b = Block(shape)
        self.A = Parameter(shape)
        self.B = Parameter(shape)
        self.C = Parameter(shape)
        self.shape = shape

    def call(self, input):
        out = self.b(input)
        out = out + self.A + self.B
        return out


#from nnabla_nas.contrib.darts import SearchNet
#net = SearchNet(in_channels=3, init_channels=16, num_cells=3, num_classes=10)

path = 'log/tensorboard'

w = FileWriter(path)

net = MyModule([5, 5])
#x = nn.Variable([1, 3, 32, 32])
x = nn.Variable([5, 5])


# print(net)
visitor = GraphVisitor(net, x)

out = net(x)
out.visit(visitor)


g = GraphDef(node=visitor._graph, versions=VersionDef(producer=22))

event = event_pb2.Event(graph_def=g.SerializeToString())
w.add_event(event)

w.close()


#import pdb; pdb.set_trace()


#v1 = nn.Variable((1,))
#v2 = nn.Variable((1,))
#v3 = nn.Variable((1,))

#print(hash(v1), hash(v2), hash(v3))

#out = v1 + v2 + v3


# shape = (1, 3, 32, 32)

# nodes = [
#     node_proto(name='A',
#                op='Variable',
#                outputsize=[shape],
#                attributes=f'shape={shape}'),
#     node_proto(name='B',
#                op='Parameter',
#                outputsize=[(3, 2, 1)],
#                attributes=f'shape={shape}'),
#     node_proto(name='Add2',
#                op='Add2',
#                input=['A', 'B'],
#                outputsize=[shape, (4, 5, 6)],
#                attributes=f'shape={shape}'),
#     node_proto(name='C',
#                op='Variable',
#                input=['Add2'],
#                outputsize=[shape],
#                attributes=f'shape={shape}'),
#     node_proto(name='D',
#                op='Variable',
#                input=['Add2'],
#                outputsize=[shape],
#                attributes=f'shape={shape}')
# ]


# g = GraphDef(node=nodes, versions=VersionDef(producer=22))

# event = event_pb2.Event(graph_def=g.SerializeToString())
# w.add_event(event)

# w.close()
