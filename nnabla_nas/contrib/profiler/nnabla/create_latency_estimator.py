import argparse
from collections import defaultdict
import numpy as np
import json
import logging

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.utils.profiler import GraphProfiler

from nnabla_nas.contrib.profiler.helpers import uid, get_search_net, get_sampled_modules

from mako.template import Template


def main(args):
    nn.logger.setLevel(logging.ERROR)

    # SearchNet
    with open(args.search_net_config) as fp:
        config = json.load(fp)
    net_config = config['network'].copy()
    net = get_search_net(net_config, "sample")

    # Latency Table
    with open(args.latency_table_json) as fp:
        latency_table = json.load(fp)

    # Compute sample latency for each runtime and sampled network
    runtimes = ["cpu:float", "cudnn:float"]

    datasets = defaultdict(list)
    runtime_scales = {}
    runtime_biases = {}
    for runtime in runtimes:
        print("Creating latency estimator of runtime = {}".format(runtime))
        for i in range(args.num_trials):
            # Context
            context, type_config = runtime.split(":")
            ctx = get_extension_context(context, type_config=type_config, device_id=args.device_id)
            nn.set_default_context(ctx)

            # Sample
            inp = nn.Variable([1] + config["input_shape"])
            out = net(inp)
            modules = get_sampled_modules(net)

            # Accum latency
            accum_latency = 0.0
            for m in modules:
                accum_latency += latency_table[uid(m)][runtime]

            # Sample latency
            if out.parent is None:
                continue
            try:
                runner = GraphProfiler(out,
                                       device_id=args.device_id,
                                       ext_name=context,
                                       n_run=args.n_run)
                runner.run()
                sampled_latency = float(runner.result["forward_all"])
                datasets[runtime].append([accum_latency, sampled_latency])
            except Exception as e:
                nn.logger.error("Measurement fails.")
                nn.logger.error(e)

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

        # Check difference bitween estimator and sampled latency
        ye = scale * x + bias
        diff = np.abs(y - ye)
        print("Runtime = {} [{} sec]".format(runtime, args.time_scale))
        print("Ave", "Std", "Min", "Max", "of the abslute error b/w target and linear regression")
        print(np.mean(diff), np.std(diff), np.min(diff), np.max(diff))

    # Read template and create python script
    with open("estimator.py.tmpl") as fp:
        script = Template(fp.read()).render(runtime_scales=runtime_scales,
                                            runtime_biases=runtime_biases)
    with open("estimator.py", "w") as fp:
        fp.write(script)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Latency Table Generation for NAS")
    parser.add_argument('--search-net-config', type=str, required=True,
                        help='Path to SearchNet jsonconfig file.')

    parser.add_argument('--device-id', type=str, help='', default="0")
    parser.add_argument('--n-run', type=int, help='Number of inputs for measurement', default=100)
    parser.add_argument('--time-scale', type=str, help='m:milli sec, u: micro sec, n: nano sec',
                        default="m", choices=["m", "u", "n"])

    parser.add_argument('--latency-table-json', type=str, help='', required=True)
    parser.add_argument('--num-trials', type=int, default=50, help='')
    args = parser.parse_args()

    main(args)
