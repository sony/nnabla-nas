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
    r"""A visitor for NNabla Graph.

    Args:
        model (Module): A NNabla Module.

    Note: All arguments in `args` should be NNabla Variables, which will be used during the `call` method.
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

    def _add_node(self, node):
        r"""Adds a new node to the graph."""
        self._graph.append(node)

    def _create_node(self, p):
        r"""Create a node given its parameter."""
        name = self._get_node_name(p)
        if (name not in self._visited) and (name not in EXCLUDED_NODES):
            self._visited.add(name)
            # add a new node to the graph
            self._add_node(
                node_proto(
                    name=name, op='Variable',
                    output_shapes=[p.shape],
                    need_grad=p.need_grad
                )
            )

    def _get_scope_from(self, scopes):
        r"""Return the scope from a list of scopes."""
        if np.all([self._on_flow[s] for s in scopes]):
            return '/'.join(scopes[0].split('/')[:-1])
        scopes = [s.split('/')[:-1] for s in scopes if not self._on_flow[s]]
        # find lowest common ancestors
        n = np.min([len(s) for s in scopes])
        scope = []
        for j in range(n):
            if np.any([scopes[0][j] != scopes[i][j] for i in range(len(scopes))]):
                break
            scope.append(scopes[0][j])
        return '/'.join(scope)

    def _get_node_name(self, p):
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
            self._create_node(ni)
            node_name = self._get_node_name(ni)
            if node_name not in EXCLUDED_NODES:
                inputs.append(node_name)

        # get function node
        self._scope[hash(f)] = self._get_scope_from(inputs)
        self._scope[hash(f)] += ('/' if self._scope[hash(f)] else '') + str(f)
        name = self._get_node_name(f)

        # make sure all outputs refer to the same function node
        outputsize = []
        for no in f.outputs:
            ho = hash(no)
            if self._scope[ho] == 'Output':
                self._visited.add(self._get_node_name(no))
                self._add_node(
                    node_proto(
                        name=self._get_node_name(no), op='Output',
                        inputs=[name],
                        need_grad=no.need_grad
                    )
                )

            self._scope[ho] = self._scope[hash(f)]
            self._tb_name[ho] = name
            self._on_flow[name] = True
            outputsize.append(no.shape)

        # add the function node
        self._visited.add(name)
        self._add_node(
            node_proto(
                name=name, op=str(f),
                inputs=inputs if len(inputs) > 0 else None,
                output_shapes=outputsize,
                need_grad=f.need_grad,
                info=f.info.args
            )
        )
