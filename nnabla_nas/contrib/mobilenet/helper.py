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

import graphviz
import imageio

default_style = {
    'fontcolor': 'white',
    'style': 'filled',
    'labelfontsize': '11',
    'shape': 'box',
    'height': '0.2'
}


def get_color(label):
    if '3x3' in label:
        return 'yellowgreen'
    if '5x5' in label:
        return 'darkorange'
    return 'brown'


def get_width(label):
    if 'MB1' in label:
        return '1'
    if 'MB3' in label:
        return '1.5'
    return '2'


def plot_mobilenet(model, filename):
    r"""Plot the architecture of MobileNet V2

    Args:
        model (Model): The search space.
    """
    features = model.modules['_features']
    candidates = model._candidates

    gr = graphviz.Digraph()
    gr.attr(landscape='True')

    # plot input
    k = 0
    gr.node('Input', shape='invhouse', fillcolor='firebrick',
            penwidth='1.5', style='filled', fontsize='11',
            fontcolor='white', width='0.75')
    gr.edge('Input', 'Conv 3x3', fontsize='8',
            label=str(features[k].input_shapes[0]))

    # plot basic blocks
    k = k + 1
    gr.node('Conv 3x3', fillcolor='aquamarine4',
            penwidth='1.5', **default_style)
    gr.edge('Conv 3x3', 'MB1 3x3', fontsize='8',
            label=str(features[k].input_shapes[0]))

    gr.node('MB1 3x3', fillcolor=get_color(
        'MB1 3x3'), penwidth='1.5', **default_style)
    prev_op = 'MB1 3x3'

    # plot the remaining blocks
    for c, n, s in model._settings:
        for i in range(n):
            k = k + 1
            selected_idx = features[k]._mixed._active
            if selected_idx < len(candidates):
                op = candidates[selected_idx]
                gr.node(op + str(k), label=op,
                        fillcolor=get_color(op),
                        width=get_width(op),
                        penwidth='0',
                        **default_style)
                gr.edge(prev_op, op + str(k), fontsize='8',
                        label=str(features[k].input_shapes[0]))
                prev_op = op + str(k)

    # last layer
    k = k + 1
    gr.node('Conv 1x1', fillcolor='aquamarine4',
            penwidth='1.5', **default_style)
    gr.edge(prev_op, 'Conv 1x1', fontsize='8',
            label=str(features[k].input_shapes[0]))

    # plot output
    prev_op = 'Conv 1x1'
    gr.node('Output', shape='house', fillcolor='deepskyblue', width='0.75',
            penwidth='1.5', style='filled', fontsize='11', fontcolor='white')
    gr.edge(prev_op, 'Output', fontsize='8',
            label=str(model._classifier.input_shapes[0]))

    gr.render(filename, view=False, cleanup=True, format='png')
    return imageio.imread(filename + '.png').transpose((2, 0, 1))
