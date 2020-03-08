from collections import OrderedDict, defaultdict

import nnabla as nn
import numpy as np

from .proto_graph import node_proto

EXCLUDED_NODES = ['CONSTANT']


def _get_unique_params(params):
    r"""Returns a dictionary with unique parameters."""
    mark = set()
    unique_params = OrderedDict()
    for k, p in params.items():
        h = hash(p)
        if h not in mark:
            mark.add(h)
            unique_params[k] = p
    return unique_params


class GraphVisitor(object):
    r"""A visitor for nnabla graph.

    Args:
        model (Module): A nnabla module.

    Note: all arguments in `args` should be nnabla variables, which will be used during the `call` method.
    """

    def __init__(self, model, *args, **kargs):
        # this will store the tensorboard name for each parameter.
        self._tb_name = dict()
        # wether the node was already constructed
        self._visited = set()
        # the current graph
        self._graph = []
        # counter for the node having same name
        self._counter = defaultdict(int)
        self._on_flow = defaultdict(bool)

        # build the nnabla graph
        outputs = model(*args, **kargs)

        # get parameters
        params = model.get_parameters(grad_only=False)
        params = _get_unique_params(params)

        # define the scope for all known parameters
        prefix = type(model).__name__ + '/'
        self._scope = defaultdict(str, {hash(p): (prefix + k) for k, p in params.items()})

        for ni in args:
            self._scope[hash(ni)] = 'Input'
            self._on_flow['Input'] = True

        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]

        # add connections to outputs
        for no in outputs:
            if isinstance(no, nn.Variable):
                self._scope[hash(no)] = 'Output'
                no.visit(self)

    def add_node(self, node):
        r"""Adds a new node to the graph."""
        self._graph.append(node)

    def create_node(self, p):
        r"""Create a node given its parameter."""
        name = self.get_node_name(p)
        if (name not in self._visited) and (name not in EXCLUDED_NODES):
            # add a new node to the graph
            self.add_node(
                node_proto(
                    name=name, op='Variable',
                    output_shapes=[p.shape],
                    attributes=f'need_grad={p.need_grad}'
                )
            )

    def get_scope_from(self, scopes):
        r"""Return the scope from a list of scopes."""
        if np.all([self._on_flow[s] for s in scopes]):
            return '/'.join(scopes[0].split('/')[:-1])
        # get scopes for each name
        scopes = [s.split('/')[:-1] for s in scopes if not self._on_flow[s]]
        n, m = np.min([len(s) for s in scopes]), len(scopes)
        inequal = [np.any([scopes[0][l] != scopes[i][l] for i in range(m)])
                   for l in range(n)]
        idx = np.argwhere(inequal)
        idx = idx.flatten()[0] if len(idx) else m
        return '/'.join(scopes[0][:idx])

    def get_node_name(self, p):
        r"""Return the node name according to tensorboard."""
        h = hash(p)

        if h not in self._scope:
            return 'CONSTANT'

        scope = self._scope[h]
        if h not in self._tb_name:
            self._tb_name[h] = scope
            if self._counter[scope]:
                self._tb_name[h] += f'_{self._counter[scope]}'
            self._counter[scope] += 1
        return self._tb_name[h]

    def __call__(self, f):
        inputs = []
        for ni in f.inputs:
            self.create_node(ni)
            node_name = self.get_node_name(ni)
            if node_name not in EXCLUDED_NODES:
                inputs.append(node_name)

        # get function node
        print('Calculate for function', f, hash(f))
        self._scope[hash(f)] = self.get_scope_from(inputs)
        self._scope[hash(f)] += ('/' if self._scope[hash(f)] else '') + str(f)
        name = self.get_node_name(f)
        print('final', name)

        # make sure all outputs refer to the same function node
        outputsize = []
        for no in f.outputs:
            ho = hash(no)
            if self._scope[ho] == 'Output':
                self.add_node(
                    node_proto(
                        name=self.get_node_name(no), op='Output',
                        inputs=[name],
                        attributes=f'need_grad={no.need_grad}'
                    )
                )
            self._scope[ho] = self._scope[hash(f)]
            self._tb_name[ho] = name
            self._on_flow[name] = True
            outputsize.append(no.shape)

        # add the function node
        self.add_node(
            node_proto(
                name=name,
                op=str(f),
                inputs=inputs if len(inputs) > 0 else None,
                output_shapes=outputsize
            )
        )
