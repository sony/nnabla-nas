import numpy as np
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from nnabla_nas.module import static as smo
import nnabla_nas.module as mo
from nnabla_nas.contrib.model import Model 

class RandomModule(smo.Graph):
    def __init__(parents, name='', channels):
        smo.Graph.__init__(self,
                           parents=parents,
                           name=name)
        self._channels = channels
        shapes = [(list(ii.shape) + 4 * [1])[:4] for ii in self.parents]
        min_shape = np.min(np.array(shapes), axis=0)
        self._shape_adaptation = {i: np.array(si[2:]) / min_shape[2:]
                                  for i, si in enumerate(shapes)
                                  if tuple(si[2:]) != tuple(min_shape[2:])}           
        projected_inputs = []
        # add an input convolution to project to the correct #channels
        for pi in self.parents:
            self.append(smo.Conv(name='{}/input_conv'.format(self.name),
                                 parents=[input_con],
                                 in_channels=pi.shape[1],
                                 out_channels=self._channels,
                                 kernel=(1, 1)))
            self.append(smo.BatchNormalization(name='{}/input_conv_bn'.format(
                                               self.name),
                                               parents=[self[-1]],
                                               n_dims=4,
                                               n_features=self._channels))
            self.append(smo.ReLU(name='{}/input_conv/relu'.format(self.name),
                                 parents=[self[-1]])) 

            projected_inputs.append(self[-1])

        for i, pii in enumerate(projected_inputs):
            if i in self._shape_adaptation:
                self.append(smo.MaxPool(name='{}/shape_adapt'
                                        '_pool_{}'.format(self.name, i),
                                         parents=[pii],
                                         kernel=self._shape_adaptation[i],
                                         stride=self._shape_adaptation[i]))
                 projected_inputs[i] = self[-1]


class Conv3x3(RandomModule):
    pass


class Conv5x5(RandomModule):
    pass


class MaxPool(RandomModule):
    pass


class AvgPool(RandomModule):
    pass




def random_graph(n_vertices, input_channels,
                 output_channels, candidates=[],
                 min_channels=32, max_channels=512):

    graph = nx.watts_strogatz_graph(n_vertices, 3, 0.5)

    # 1. make the graph directed, such that it is not cyclic
    G = nx.DiGraph()
    G.name = graph.name
    G.add_nodes_from(graph)
    G.add_edges_from(((u, v, deepcopy(data))
            for u, nbrs in graph.adjacency()
            for v, data in nbrs.items()
            if v > u))
    G.graph = deepcopy(graph.graph)


    # 2. add a single input and output to the network
    G.add_node(-1) # input
    G.add_node(n_vertices+1) # output
    adj_matrix = nx.adjacency_matrix(G).todense()
    inputs = np.where(np.ravel(np.sum(adj_matrix, axis=0) == 0))
    outputs = np.where(np.ravel(np.sum(adj_matrix, axis=1) == 0))
    for i in inputs[0]:
        G.add_edge(-1, i)

    for o in outputs[0]:
        G.add_edge(o, n_vertices+1)

    #3. randomly choose the layer type and the channel count
    for i in range(n_vertices+2):
        random_classes = {i-1: np.random.randint(0, len(candidates), 1)}
        random_channels = {i-1: np.random.randint(min_channels, max_channels,  1)}

    nx.set_node_attributes(G=G, name='class', values=random_classes)
    nx.set_node_attributes(G=G, name='channels', values=random_channels)
    return G


class TrainNet(Model, smo.Graph):
    def __init__(self, n_vertices, input_channels,
                 output_channels, candidates, min_channels=32,
                 max_channels=512):
        # 1. draw a random network graph
        g = random_dnn(n_vertices,
                       input_channels,
                       output_channels,
                       candidates,
                       min_channels,
                       max_channels)

        # 2. convert it to an actual network
        # TODO

    


n_vertices = 5 
modules = {0: 'Conv3x3',
           1: 'Conv5x5',
           2: 'MaxPool',
           3: 'AvgPool',
           4: 'GavgPool'}

g = random_dnn(n_vertices=n_vertices,
               input_channels=3,
               output_channels=512,
               candidates=modules,
               min_channels=32,
               max_channels=512)
import pdb; pdb.set_trace()
random_classes  = {}
random_channels = {}


plt.figure(1)
pos = nx.spring_layout(g)
nx.draw(g, pos=pos)
nx.draw_networkx_labels(g, pos=pos)
plt.show()
