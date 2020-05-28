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
import os
import glob
import numpy as np
import json
import logging
from collections import defaultdict

import nnabla as nn

from mako.template import Template


def main(args):
    nn.logger.setLevel(logging.ERROR)

    accum_latency_files = glob.glob(os.path.join(args.accum_latency, "*.json"))
    accum_latency_files = sorted(accum_latency_files)

    latency_files = glob.glob(os.path.join(args.sampled_latency, "*", "latest_results", "*.json"))
    latency_files = sorted(latency_files)

    datasets = defaultdict(list)
    with open(accum_latency_files[0]) as fp:
        runtimes = json.load(fp).keys()
    runtime_scales = {}
    runtime_biases = {}
    for runtime in runtimes:
        for i in range(len(latency_files)):
            with open(latency_files[i]) as fp:
                sampled_latency = json.load(fp)
            with open(accum_latency_files[i]) as fp:
                accum_latency = json.load(fp)
            accum_latency_ = accum_latency[runtime]
            sampled_latency_ = sampled_latency["Execution_Data"][runtime][args.inference_key]["Avg_Time"]
            datasets[runtime].append([accum_latency_, sampled_latency_])

        # Linear Regression
        xy = np.asarray(datasets[runtime])
        x = xy[:, 0]
        y = xy[:, 1]
        mx = np.mean(x)
        my = np.mean(y)
        scale = np.sum((x - mx) * (y - mx)) / np.sum((x - mx) ** 2.0)
        bias = my - scale * mx
        runtime_scales[runtime] = scale
        runtime_biases[runtime] = bias

        # Check difference between estimator and sampled latency
        ye = scale * x + bias
        gap = np.abs(y - ye) / y * 100
        print("Runtime = {} [usec]".format(runtime))
        print("Ave", "Std", "Min", "Max", "of (the abslute error b/w target and linear regression) / target")
        print(np.mean(gap), np.std(gap), np.min(gap), np.max(gap))

    # Read template and create python script
    with open("estimator.py.tmpl") as fp:
        script = Template(fp.read()).render(runtime_scales=runtime_scales,
                                            runtime_biases=runtime_biases)
    with open("estimator.py", "w") as fp:
        fp.write(script)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create the latency estimator for SNPE")
    parser.add_argument('--accum-latency', type=str, required=True,
                        help='Directory to the accumulated latency files.')
    parser.add_argument('--sampled-latency', type=str, required=True,
                        help='Directory to the sampled latency files.')
    parser.add_argument('--inference-key', type=str, help='', default="Total Inference Time",
                        choices=["Forward Propagate",
                                 "Total Inference Time"])

    args = parser.parse_args()
    main(args)
