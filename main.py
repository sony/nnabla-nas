import argparse
import json

import nnabla as nn
from nnabla.ext_utils import get_extension_context

from nnabla_nas.contrib import Darts, NetworkCIFAR
from nnabla_nas.runner import Searcher, Trainer


def search(model, config):
    return Searcher(model, config)


def train(model, config):
    return Trainer(model, config)


def pass_args(parser):
    # model-parameter-related
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension path. ex) cpu, cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='1',
                        help='Device ID the training run on. \
                        This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--num-cells", type=int, default=8)
    parser.add_argument("--num-nodes", type=int, default=4,
                        help='Number of nodes per cell, must be more than 2.')
    parser.add_argument("--init-channels", type=int, default=16,
                        help='Number of output filters of CNN, must be even.')
    parser.add_argument("--mode", type=str, default='sample')
    parser.add_argument("--shared-params", action='store_true')
    parser.add_argument('--config-file', type=str,
                        default=None, help='config file')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # search the architecture
    search_parser = subparsers.add_parser('search')
    search_parser.set_defaults(func=search)
    pass_args(search_parser)
    search_parser.add_argument("--mini-batch-size", type=int, default=8)

    # train the model
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    pass_args(train_parser)
    train_parser.add_argument("--batch-size-train", type=int, default=8)
    train_parser.add_argument("--batch-size-valid", type=int, default=8)
    train_parser.add_argument('--drop-path-prob', type=float, default=0.2)
    train_parser.add_argument('--auxiliary-weight', type=float, default=0.4)
    train_parser.add_argument('--auxiliary', action='store_true',
                              default=False, help='use auxiliary tower')
    train_parser.add_argument('--cutout', action='store_true',
                              default=False, help='use cutout')
    train_parser.add_argument('--cutout-length', type=int, default=16,
                              help='Cutout length')
    args = parser.parse_args()

    if args.config_file is not None:
        config = json.load(open(args.config_file))
        config.update(vars(args))

    # setup context for nnabla
    ctx = get_extension_context(
        config['context'], device_id=config['device_id'])
    nn.set_default_context(ctx)

    if args.func == search:
        model = Darts(
            shape=(args.mini_batch_size, 3, 32, 32),
            init_channels=args.init_channels,
            num_cells=args.num_cells,
            num_choices=args.num_nodes,
            num_classes=10,
            shared_params=args.shared_params,
            mode=args.mode
        )
        args.func(model, config).run()
    else:
        genotype = json.load(open(config['arch']+'.json'))
        # this code only work for shared params
        assert config['shared_params']
        train(
            model=NetworkCIFAR(
                shape=(args.batch_size_train, 3, 32, 32),
                init_channels=args.init_channels,
                num_cells=args.num_cells,
                num_classes=10,
                auxiliary=args.auxiliary,
                genotype=genotype
            ),
            config=config
        ).run()
