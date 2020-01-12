import argparse
import json

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
    parser.add_argument("--minibatch-size", type=int, default=8)
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

    # train the model
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    pass_args(train_parser)
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

    if args.func == search:
        model = Darts(
            shape=(args.minibatch_size, 3, 32, 32),
            init_channels=args.init_channels,
            num_cells=args.num_cells,
            num_choices=args.num_nodes,
            num_classes=10,
            shared_params=args.shared_params,
            mode=args.mode
        )
    else:
        model = NetworkCIFAR(
            shape=(args.minibatch_size, 3, 32, 32),
            init_channels=args.init_channels,
            num_cells=args.num_cells,
            num_choices=args.num_nodes,
            num_classes=10,
            shared_params=args.shared_params,
            mode=args.mode,
            drop_prob=args.drop_path_prob,
            auxiliary=args.auxiliary
        )

    args.func(model, config).run()
