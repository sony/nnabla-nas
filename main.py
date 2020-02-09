import argparse
import json

import nnabla as nn
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context
from nnabla.logger import logger

import args
import nnabla_nas.contrib as contrib
from nnabla_nas import runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension path. ex) cpu, cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='1',
                        help='Device ID the training run on. \
                        This is only valid if you specify `-c cudnn`.')
    parser.add_argument('--search', '-s', action='store_true',
                        help='config file')
    parser.add_argument('--algorithm', '-a', type=str, default='DartsSeacher',
                        choices=runner.__all__, help='Algorithm used to run')
    parser.add_argument('--config-file', '-f', type=str, help='config file',
                        default=None)
    parser.add_argument('--output-path', '-o', type=str, help='config file',
                        default=None)

    options = parser.parse_args()

    config = json.load(open(options.config_file)) if options.config_file \
        else dict()
    config.update(vars(options))

    # setup context for nnabla
    ctx = get_extension_context(
        config['context'],
        device_id=config['device_id']
    )
    nn.set_default_context(ctx)

    # build the model
    attributes = config['network'].copy()
    algorithm = contrib.__dict__[attributes.pop('name')]

    model = algorithm.SearchNet(**attributes) if config['search'] else \
        algorithm.TrainNet(**attributes)

    options = args.Configuration(config)

    # define contraints
    regularizer = args.RegularizerParser(options).parse(config)

    # define dataloader for training and validating
    dataloader = args.DataloaderParser(options).parse(config)

    # define optimizer
    max_iter = (len(dataloader['train']) * options.epoch
                // options.mbs_train)
    opt_parser = args.OptimizerParser(options, max_iter=max_iter)

    optimizer = opt_parser.parse(config.get('optimizer', dict()))

    # a placeholder to store input and output variables
    placeholder = args.PlaceholderParser(options).parse(config)

    logger.info('Configurations:\n' + options.summary())

    runner.__dict__[config['algorithm']](
        model,
        placeholder=placeholder,
        optimizer=optimizer,
        dataloader=dataloader,
        regularizer=regularizer,
        criteria=lambda o, t: F.mean(F.softmax_cross_entropy(o, t)),
        evaluate=lambda o, t: F.mean(F.top_n_error(o, t)),
        args=options
    ).run()
