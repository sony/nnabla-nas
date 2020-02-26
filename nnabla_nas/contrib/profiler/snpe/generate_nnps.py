import argparse
import os
import shutil
import logging

import nnabla as nn


from nnabla_nas.contrib.profiler.helpers import (create_parameters, nnp_save,
                                                 get_unique_modules, get_search_net)


def main(args):
    nn.logger.setLevel(logging.ERROR)

    # Unique modules
    net = get_search_net(args.search_net, num_classes=args.num_classes, mode=args.mode)
    inp = nn.Variable([1, args.in_channels, args.in_height, args.in_width])
    out = net(inp)
    unique_mods = get_unique_modules(net)

    if os.path.exists(args.nnp_dir):
        shutil.rmtree(args.nnp_dir)
    os.makedirs(args.nnp_dir)

    # Generate NNPs
    for mod_name, umod in unique_mods.items():
        path = "{}/{}.nnp".format(args.nnp_dir, mod_name)
        inp = [nn.Variable(shape) for shape in umod.input_shapes]
        out = umod(*inp)
        if out.parent is None:
            continue
        nn.parameter.clear_parameters()
        create_parameters(out)
        nnp_save(path, mod_name, inp, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NNP Files Generation for NAS.")
    parser.add_argument('--in-channels', type=int, default=3, help='')
    parser.add_argument('--in-height', type=int, default=32, help='')
    parser.add_argument('--in-width', type=int, default=32, help='')
    parser.add_argument('--init-channels', type=int, default=36, help='')
    parser.add_argument('--num-cells', type=int, default=15, help='')
    parser.add_argument('--num-classes', type=int, default=10, help='')

    parser.add_argument('--nnp-dir', type=str, required=True, help='Directory for NNP input files.')
    parser.add_argument('--search-net', type=str, help='Name of SearchNet.', required=True)
    parser.add_argument('--mode', type=str, default="full", help='Mode of SearchNet')
    args = parser.parse_args()

    main(args)
