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

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F


from ..... import module as Mo
from .modules.dynamic_op import DynamicBatchNorm
from ....common.ofa.utils.random_resize_crop import OFAResize


def set_running_statistics(model, dataloader, dataloader_batch_size, data_size, batch_size, inp_shape):
    """Re-calculates batch normalization mean and variance"""

    logger.info('Start batch normalization stats calibaration')

    model.apply(training=False)

    for name, m in model.get_modules():
        if isinstance(m, Mo.BatchNormalization):
            m.training = True
            m.set_running_statistics = True
            m.mean_est.reset()
            m.var_est.reset()
        elif isinstance(m, DynamicBatchNorm):
            m.set_running_statistics = True

    def load_data(placeholder, data):
        inp_list = []
        for inp, x in zip(placeholder, data):
            if isinstance(x, nn.NdArray):
                inp.data = x
            else:
                inp.d = x
            inp_list.append(inp)
        return inp_list

    with nn.no_grad():
        resize = OFAResize()
        transform = dataloader.transform('valid')
        with nn.auto_forward(True):
            # Note: only support NCHW data format
            x = [nn.Variable([dataloader_batch_size] + shape) for shape in inp_shape]
            accum = batch_size // dataloader_batch_size + 1
            for i in range(data_size // batch_size):
                if accum > 1:
                    x_accum = []
                    for _ in range(accum):
                        data = dataloader.next()
                        x_accum.append(load_data(x, data['inputs']))
                    x_concat = [F.concatenate(*[x_accum[i][j] for i in range(len(x_accum))], axis=0)
                                for j in range(len(x))]
                else:
                    data = dataloader.next()
                    x_concat = load_data(x, data['inputs'])
                inputs = [resize(transform(x[:batch_size, :, :, :])) for x in x_concat]
                model(*inputs)

    for name, m in model.get_modules():
        if isinstance(m, Mo.BatchNormalization):
            if m.mean_est.count > 0:
                feature_dim = m.mean_est.avg.shape[1]
                new_mean = np.concatenate(
                    (m.mean_est.avg, m._mean.d[:, feature_dim:, :, :]), axis=1)
                new_var = np.concatenate(
                    (m.var_est.avg, m._var.d[:, feature_dim:, :, :]), axis=1)
                m._mean.d = new_mean
                m._var.d = new_var
            m.set_running_statistics = False
        elif isinstance(m, DynamicBatchNorm):
            m.set_running_statistics = False

    logger.info('Batch normalization stats calibaration finished')
