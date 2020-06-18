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

import argparse
import csv
import json
import time
import os
from pathlib import Path

import nnabla as nn
import numpy as np
from nnabla.ext_utils import get_extension_context
from nnabla.logger import logger
from nnabla.utils.profiler import GraphProfiler, convert_time_scale
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

from ... import contrib
from .estimator import Estimator


class Profiler(GraphProfiler):
    """NNabla GraphProfiler

    Args:
        graph (:class:`nnabla.Variable`): Instance of `nnabla.Variable` class. Profiler find all
            functions which compose network graph from root `nnabla.Variable` to this
            `nnabla.Variable`.
        device_id (str): gpu device id.
        ext_name (str): Extension name. e.g. 'cpu', 'cuda', 'cudnn' etc.
        solver (:class:`nnabla.solvers.Solver`): Instance of `nnabla.solvers.Solver` for optimizing
            the parameters of the computation graph. If None, the training process is ignored.
            Default value is None.
        n_run (int): This argument specifies how many times the each functions` execution time
            are measured. Default value is 100.
        max_measure_execution_time (float): Maximum time of executing measurement for each functions.
            This argument has higher priority than ``n_run``. When the measurement time for each
            functions get bigger than this argument, this class stops measuring and goes to next
            function, unless the total times of measurement are less than n_run. Default value is 1 [sec.
        time_scale (str): Time scale to display. ['m', 'u', 'n'] (which stands for 'mili', 'micro'
            and 'nano')
        n_warmup (int, optional): The number of iterations for warming up. Defaults to 10.
        outlier (float, optional): The portion of outliers which will be remove from the statistical
            results. Defaults to 0.0.
    """

    def __init__(self, graph, device_id, ext_name, solver=None, n_run=100,
                 max_measure_execution_time=1, time_scale="m",
                 n_warmup=10, outlier=0.0):
        super().__init__(graph=graph, device_id=device_id, ext_name=ext_name, solver=solver,
                         n_run=n_run, max_measure_execution_time=max_measure_execution_time)
        self.n_warmup = n_warmup
        self.outlier = outlier

    def _measure_execution_time(self, execution, *execution_args):
        runtime = []
        for i in range(self.n_run + self.n_warmup):
            self.ext_module.synchronize(device_id=self.device_id)
            start = time.perf_counter()
            execution(*execution_args)
            self.ext_module.synchronize(device_id=self.device_id)
            stop = time.perf_counter()
            if i >= self.n_warmup:
                runtime.append(stop - start)

        excluded = int(self.outlier * self.n_run)
        runtime = np.sort(runtime)
        if excluded:
            runtime = runtime[excluded:-excluded]

        mean_time = convert_time_scale(np.mean(runtime), format=self.time_scale)
        mean_time = "{:.8f}".format(mean_time)

        std_time = convert_time_scale(np.std(runtime), format=self.time_scale)
        std_time = "{:.8f}".format(std_time)

        return mean_time, std_time


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


class LatencyPredictor(Estimator):
    """An offline latency predictor.

    Args:
        data_file (str): A path to training data file.
        hidden_layer_sizes (tuple, optional): The architecture of MLPRegressor.
            Defaults to (100, 100, 100,).
    """

    def __init__(self, data_file, hidden_layer_sizes=(100, 100, 100,), **kargs):
        logger.info('Training the LatencyPredictor!')
        self._predictor = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            early_stopping=True,
            random_state=0,
            ** kargs
        )
        data = np.genfromtxt(data_file, delimiter=',')
        self._predictor.fit(data[:, :-1], data[:, -1])
        logger.info('LatencyPredictor with loss={:.5f}'.format(self._predictor.loss_))
        self._dim = data.shape[1] - 1

    def arch_repr(self, model):
        r"""Get vectorial representation of an arch.

        Args:
            model (Model): The model to get the representation.

        Returns:
            numpy.array: A vector representing the architecture.
        """
        from ...module.mixedop import MixedOp
        arch, offset = np.zeros((1, self._dim)), 0
        for _, m in model.get_modules():
            if isinstance(m, MixedOp):
                arch[0, m._active + offset] = 1.0
                offset += len(m._ops)
        return arch

    def get_estimation(self, model):
        return self._predictor.predict(self.arch_repr(model))[0]


def generate_dataset(args, model, shape):
    r"""Generate a dataset with measured latency.

    Args:
        args (dict): Configurations.
        model (Model): The search space.
        shape (tuple of int): The input shape
    """
    from ...module.mixedop import MixedOp
    model.apply(training=False)
    for _, p in model.get_parameters().items():
        p.need_grad = False

    # one-hot encoding writer
    ohe_file = open(args.output, 'w')
    label_writer = csv.writer(ohe_file)

    choices = [m for _, m in model.get_modules() if isinstance(m, MixedOp)]
    tensor_size = sum([len(m._ops) for m in choices]) + 1

    for run in tqdm(range(args.n_samples)):
        # sample an architecture
        x = nn.Variable.from_numpy_array(np.zeros([args.batch_size] + shape))
        out = model(x)

        rep, offset = np.zeros(tensor_size), 0
        for m in choices:
            rep[m._active + offset] = 1.0
            offset += len(m._ops)

        # estimate the graph profiler
        runner = Profiler(
            out,
            device_id=args.device_id,
            ext_name=args.context,
            time_scale=args.time_scale,
            n_run=args.n_run
        )
        # profiling each op
        runner.time_profiling_forward()
        # profiling the whole graphs
        mean_time, std_time = runner._measure_execution_time(out.forward)
        mean_time, std_time = float(mean_time), float(std_time)

        # print output of one-hot encoding
        rep[-1:] = mean_time
        label_writer.writerow(rep)
        ohe_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', '-c', type=str, default='cudnn')
    parser.add_argument('--device-id', '-d', type=str, default='1')
    parser.add_argument('--time-scale', '-t', type=str, default='m')
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--output', '-o', type=str, default='log/latency.txt')
    parser.add_argument('--n-samples', '-n', type=int, default=10000)
    parser.add_argument('--n-run', type=int, default=100)
    parser.add_argument('--config-file', '-f', type=str,
                        help=('The config file containing parameters of'
                              'the search space'), default=None)

    args = parser.parse_args()

    # setup the context
    ctx = get_extension_context(
        ext_name=args.context,
        device_id=args.device_id
    )
    nn.set_default_context(ctx)

    config = json.load(open(args.config_file))
    shape = config['input_shape']

    # build the model
    attributes = config['network'].copy()
    algorithm = contrib.__dict__[attributes.pop('search_space')]
    model = algorithm.SearchNet(**attributes)

    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    generate_dataset(args, model, shape)
