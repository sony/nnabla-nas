import argparse
import os
import shutil
import json
import logging
from collections import defaultdict

import nnabla as nn

from nnabla_nas.contrib.profiler.helpers import create_parameters, nnp_save, get_search_net
from nnabla_nas.contrib.profiler.helpers import uid, get_sampled_modules


def main(args):
    nn.logger.setLevel(logging.ERROR)

    # Result NNPs
    if os.path.exists(args.nnp_dir):
        shutil.rmtree(args.nnp_dir)
    os.makedirs(args.nnp_dir)

    # Result Accum Latency
    accum_latency_dir = "{}-accum-latency".format(args.nnp_dir)
    if os.path.exists(accum_latency_dir):
        shutil.rmtree(accum_latency_dir)
    os.makedirs(accum_latency_dir)

    # Latency Table
    with open(args.latency_table_json) as fp:
        latency_table = json.load(fp)

    # SearchNet
    net = get_search_net(args.search_net, num_classes=args.num_classes, mode=args.mode)

    # Generate sampled nnp and save its accumulated latency
    accum_latencies = []
    for i in range(args.num_trials):
        print("Accumulated Latency at {}".format(i))

        # Sample network
        inp = nn.Variable([1, args.in_channels, args.in_height, args.in_width])
        out = net(inp)

        # DEBUG
        # print(unique_mods)

        # Save NNP
        nn.parameter.clear_parameters()
        create_parameters(out)
        nnp_file = os.path.join(args.nnp_dir, "sampled-net-{:03d}.nnp".format(i))
        nnp_save(nnp_file, "SampledNet", inp, out)

        # Accumulate
        runtimes = ["CPU", "GPU", "GPU_FP16", "DSP"]
        accum_latency = defaultdict(float)
        try:
            for runtime in runtimes:
                # Sample
                out = net(inp)
                modules = get_sampled_modules(net)
                # Accum latency
                accum_latency_ = 0.0
                for m in modules:
                    accum_latency_ += latency_table[uid(m)][runtime]["Layers Time"]["Avg_Time"]
                    accum_latency[runtime] = accum_latency_
        except Exception as e:
            print("Lookup fails.")
            print(e)
            raise RuntimeError("Sampled nnp contains {} "
                               "which did not appear in the generation process of the latency table."
                               .format(m))
        print(accum_latency)
        accum_latencies.append(accum_latency)

    # Save accumulated latency
    for i, accum_latency in enumerate(accum_latencies):
        with open(os.path.join(accum_latency_dir, "{:03d}.json".format(i)), "w") as fp:
            json.dump(accum_latency, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample NNPs and create accumulated latency.")
    parser.add_argument('--in-channels', type=int, default=3, help='')
    parser.add_argument('--init-channels', type=int, default=36, help='')
    parser.add_argument('--in-height', type=int, default=32, help='')
    parser.add_argument('--in-width', type=int, default=32, help='')
    parser.add_argument('--num-cells', type=int, default=15, help='')
    parser.add_argument('--num-classes', type=int, default=10, help='')
    parser.add_argument('--latency-table-json', type=str, help='', required=True)
    parser.add_argument('--num-trials', type=int, default=100, help='')
    parser.add_argument('--nnp-dir', type=str, required=True, help='Directory for NNP input files.')
    parser.add_argument('--search-net', type=str, help='Name of SearchNet.', required=True)
    parser.add_argument('--mode', type=str, default="full", help='Mode of SearchNet')
    args = parser.parse_args()

    main(args)
