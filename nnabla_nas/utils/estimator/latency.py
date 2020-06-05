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


import nnabla as nn
from nnabla.logger import logger
from nnabla.utils.profiler import GraphProfiler

from .estimator import Estimator


class LatencyEstimator(Estimator):
    """Latency estimator.

    Args:
        device_id (int): gpu device id.
        ext_name (str): Extension name. e.g. 'cpu', 'cuda', 'cudnn' etc.
        n_run (int): This argument specifies how many times the each functions
            execution time are measured. Default value is 10.
        weight (float, optional): Weight used in the reinforce algorithm.
        bound (float, optional): Maximum bound used in the reinforce algorithm.
    """

    def __init__(self, device_id=None, ext_name=None, n_run=10, weight=0.1, bound=5):
        ctx = nn.context.get_current_context()
        if device_id is None:
            device_id = int(ctx.device_id)
        if ext_name is None:
            ext_name = ctx.backend[0].split(':')[0]
        self._device_id = device_id
        self._ext_name = ext_name
        self._n_run = n_run
        self._weight = weight
        self._bound = bound

    def predict(self, module):
        idm = str(module)
        if idm not in self.memo:
            self.memo[idm] = dict()
        mem = self.memo[idm]
        key = '-'.join([str(k[1:]) for k in module.input_shapes])

        if key not in mem:
            state = module.training
            module.apply(training=False)  # turn off training
            try:
                # run profiler
                nnabla_vars = [nn.Variable((1,) + s[1:])
                               for s in module.input_shapes]
                runner = GraphProfiler(module.call(*nnabla_vars),
                                       device_id=self._device_id,
                                       ext_name=self._ext_name,
                                       n_run=self._n_run)
                runner.run()
                latency = float(runner.result['forward_all'])
            except Exception as err:
                latency = 0
                logger.warning(f'Latency calculation fails: {idm}[{key}]')
                logger.warning(str(err))

            mem[key] = latency
            module.apply(training=state)  # recover training state
        return mem[key]
