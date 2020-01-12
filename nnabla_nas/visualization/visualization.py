import sys
from graphviz import Digraph
import json
import os

NORMAL_OPS = [
    'dil_conv_3x3',
    'dil_conv_5x5',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'max_pool_3x3',
    'avg_pool_3x3',
    'identity',
    'zero'
]
REDUCE_OPS = [
    'dil_conv_3x3',
    'dil_conv_5x5',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'max_pool_3x3',
    'avg_pool_3x3',
    'fac_reduce',
    'zero'
]


def plot(choice, prob, num_choices, ops, filename):
    g = Digraph(format='png',
                edge_attr=dict(fontsize='14', fontname="times"),
                node_attr=dict(style='filled', shape='rect', align='center', fontsize='20',
                               height='0.5', width='0.5', penwidth='2', fontname="times"),
                engine='dot')
    g.body.extend(['rankdir=LR'])

    # plot vertices
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    g.node("c_{k}", fillcolor='palegoldenrod')
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
            g.edge(u, v, label='<{:.3f}> '.format(p)+ops[t], fillcolor="gray")

    g.render(filename, view=False, cleanup=True)


def visualize(arch_file, num_choices, path):
    conf = json.load(open(arch_file))
    for name in ['reduce', 'normal']:
        plot(choice=conf[name + '_alpha'], 
            prob=conf[name + '_prob'], 
            num_choices=num_choices,
            ops=REDUCE_OPS if name == 'reduce' else NORMAL_OPS,
            filename=os.path.join(path, name)
        )
