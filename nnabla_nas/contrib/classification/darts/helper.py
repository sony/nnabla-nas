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

import json
import os

from graphviz import Digraph
import imageio
from nnabla.logger import logger
import numpy as np
from scipy.special import softmax

from ....utils.helper import write_to_json_file
from .modules import CANDIDATES


def plot(choice, prob, filename):
    g = Digraph(format='png',
                edge_attr=dict(fontsize='14', fontname="times"),
                node_attr=dict(style='filled', shape='rect', align='center',
                               fontsize='20', height='0.5', width='0.5',
                               penwidth='2', fontname="times"),
                engine='dot')
    g.body.extend(['rankdir=LR'])

    # plot vertices
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    g.node("c_{k}", fillcolor='palegoldenrod')

    OPS = list(CANDIDATES.keys())
    num_choices = len(prob)
    for i in range(num_choices):
        g.node(str(i + 2), fillcolor='lightblue')

    # plot edges
    for i in range(num_choices):
        g.edge(str(i + 2), "c_{k}", fillcolor="gray")
    for i in range(num_choices):
        v = str(i + 2)
        for (t, node), p in zip(choice[v], prob[v]):
            if node == 0:
                u = 'c_{k-2}'
            elif node == 1:
                u = 'c_{k-1}'
            else:
                u = str(node)
            g.edge(u, v, label='<{:.3f}> '.format(p)+OPS[t], fillcolor="gray")
    g.render(filename, view=False, cleanup=True)
    return imageio.imread(filename+'.png').transpose((2, 0, 1))


def visualize(arch_file, path):
    conf = json.load(open(arch_file))
    images = dict()
    for name in ['reduce', 'normal']:
        images[name] = plot(
            choice=conf[name + '_alpha'],
            prob=conf[name + '_prob'],
            filename=os.path.join(path, name)
        )
    return images


def parse_weights(alpha, num_choices):
    offset = 0
    cell, prob, choice = dict(), dict(), dict()
    for i in range(num_choices):
        cell[i + 2], prob[i + 2] = list(), list()
        W = [softmax(alpha[j + offset].d.flatten()) for j in range(i + 2)]
        # Note: Zero Op shouldn't be included
        edges = sorted(range(i + 2), key=lambda k: -max(W[k][:-1]))
        for j, k in enumerate(edges):
            if j < 2:  # select the first two best Ops
                idx = np.argmax(W[k][:-1])
                cell[i + 2].append([int(idx), k])
                prob[i + 2].append(float(W[k][idx]))
                choice[k + offset] = int(idx)
            else:  # assign Zero Op to the rest
                choice[k + offset] = int(len(W[k]) - 1)
        offset += i + 2
    return cell, prob, choice


def save_dart_arch(model, output_path):
    r"""Saves DARTS architecture.

    Args:
        model (Model): The model.
        output_path (str): Where to save the architecture.
    """
    memo = dict()
    for name, alpha in zip(['normal', 'reduce'],
                           [model._alpha[0], model._alpha[1]]):
        for k, v in zip(['alpha', 'prob', 'choice'],
                        parse_weights(alpha, model._num_choices)):
            memo[name + '_' + k] = v
    arch_file = os.path.join(output_path, 'arch.json')
    logger.info('Saving arch to {}'.format(arch_file))
    write_to_json_file(memo, arch_file)
    visualize(arch_file, output_path)
