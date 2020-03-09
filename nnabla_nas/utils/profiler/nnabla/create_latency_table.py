import argparse
from collections import defaultdict
import json
import csv
import logging

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.utils.profiler import GraphProfiler

from nnabla_nas.utils.profiler.helpers import get_unique_modules, get_search_net


def main(args):
    nn.logger.setLevel(logging.ERROR)

    # SearchNet
    with open(args.search_net_config) as fp:
        config = json.load(fp)
    net_config = config['network'].copy()
    net = get_search_net(net_config, "full")
    inp = nn.Variable([1] + config["input_shape"])
    out = net(inp)
    unique_mods = get_unique_modules(net)

    # Compute latency for each runtime and module
    latency_table = defaultdict(lambda: defaultdict(float))
    runtimes = ["cpu:float", "cudnn:float"]
    for runtime in runtimes:
        print("Measuring latency of runtime = {}".format(runtime))
        for mod_name, umod in unique_mods.items():
            # Context
            context, type_config = runtime.split(":")
            ctx = get_extension_context(context, type_config=type_config, device_id=args.device_id)
            nn.set_default_context(ctx)
            inp = [nn.Variable(shape) for shape in umod.input_shapes]
            out = umod(*inp)
            if out.parent is None:
                continue
            try:
                runner = GraphProfiler(out,
                                       device_id=args.device_id,
                                       ext_name=context,
                                       n_run=args.n_run)
                runner.run()
                latency = float(runner.result["forward_all"])
                latency_table[mod_name][runtime] = latency
            except Exception as e:
                nn.logger.error("Measurement of the module ({}) fails.".format(mod_name))
                nn.logger.error(e)

    # Json
    with open("{}.json".format(args.table_name), "w") as fp:
        json.dump(latency_table, fp)

    # CSV
    with open("{}.csv".format(args.table_name), "w") as fp:
        writer = csv.writer(fp, delimiter=",")
        for k0 in latency_table.keys():
            for k1 in latency_table[k0].keys():
                v = latency_table[k0][k1]
                writer.writerow([k0, k1, v])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Latency Table Generation for NAS")
    parser.add_argument('--search-net-config', type=str, required=True,
                        help='Path to SearchNet jsonconfig file.')
    parser.add_argument('--table-name', type=str, help='', required=True)
    parser.add_argument('--device-id', type=str, help='', default="0")
    parser.add_argument('--n-run', type=int, help='Number of inputs for measurement', default=100)
    parser.add_argument('--time-scale', type=str, help='m:milli sec, u: micro sec, n: nano sec',
                        default="m", choices=["m", "u", "n"])
    args = parser.parse_args()

    main(args)
