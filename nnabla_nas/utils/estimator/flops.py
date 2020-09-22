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

import numpy as np
from nnabla import logger

from .estimator import Estimator


def convolution(func):
    r"""Returns the number of FLOps for a convolution."""
    C_in = func.inputs[1].shape[1]
    kernel = func.inputs[1].shape[2:]
    if func.info.args['channel_last']:
        C_in = func.inputs[1].shape[-1]
        kernel = func.inputs[1].shape[1:-1]
    out_shape = func.outputs[0].shape[func.info.args['base_axis']:]
    flops_per_filter = C_in*2*np.prod(kernel) - int(len(func.inputs) == 2)
    return flops_per_filter * np.prod(out_shape) / func.info.args['group']


def affine(func):
    r"""Returns the number of FLOps for an affine."""
    input_shape = func.inputs[1].shape
    not_use_bias = int(len(func.inputs) == 2)
    return (2 * input_shape[0] - not_use_bias) * np.prod(input_shape[1:])


def batch_normalization(func):
    r"""Returns the number of FLOPs for a batch normalization."""
    return 2*np.prod(func.inputs[0].shape[1:])


def depthwise_convolution(func):
    r"""Returns the number of FLOPs for a depthwise convolution."""
    C_in = func.inputs[1].shape[1]
    kernel = func.inputs[1].shape[1:]
    out_shape = func.outputs[0].shape[func.info.args['base_axis']:]
    not_use_bias = int(len(func.inputs) == 2)
    return C_in * (2*np.prod(kernel) - not_use_bias) * np.prod(out_shape[1:])


def elementwise_operation(func):
    r"""Returns the number of FLOPs for a basic function."""
    return np.prod(func.inputs[0].shape[1:])


def average_pooling(func):
    r"""Returns the number of FLOPs for an average pooling."""
    return np.prod(func.info.args['kernel']) * np.prod(func.outputs[0].shape[1:])


def global_average_pooling(func):
    r"""Returns the number of FLOPs for a global average pooling."""
    return np.prod(func.inputs[0].shape[1:])


def undefined_op(func):
    r"""Returns the number of FLOps for undefined operations."""
    logger.warn(f'FLOps of {func.info.type_name} were ignored. Returns 0.')
    return 0


class _Visitor():
    def __init__(self):
        self.meta = {
            'Convolution': convolution,
            'Deconvolution': convolution,

            'Affine': affine,

            'BatchNormalization': batch_normalization,
            'FusedBatchNormalization': batch_normalization,

            'DepthwiseConvolution': depthwise_convolution,
            'DepthwiseDeconvolution': depthwise_convolution,

            'AveragePooling': average_pooling,
            'GlobalAveragePooling': global_average_pooling,

            'Add2': elementwise_operation,
            'Sub2': elementwise_operation,
            'Div2': elementwise_operation,
            'Mul2': elementwise_operation,
            'Pow2': elementwise_operation,

            'AddScalar': elementwise_operation,
            'PowScalar': elementwise_operation,
            'MulScalar': elementwise_operation,

            'Sum': elementwise_operation,
            'Mean': elementwise_operation,
            'ReLU': elementwise_operation,
            'SELU': elementwise_operation,
            'Tanh': elementwise_operation,
            'LeakyReLU': elementwise_operation,
            'ReLU6': elementwise_operation,
            'Sigmoid': elementwise_operation,
        }

        self.reset()

    def reset(self):
        self._flops = 0

    def __call__(self, func):
        self._flops += self.meta.get(func.info.type_name, undefined_op)(func)


class FLOPsEstimator(Estimator):
    r"""Estimator for the number of FLOPs."""

    def __init__(self):
        self.visitor = _Visitor()

    def predict(self, model):
        self.visitor.reset()
        model.visit(self.visitor)
        return self.visitor._flops
