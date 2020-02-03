import argparse
import json

import nnabla as nn
from nnabla.ext_utils import get_extension_context

import nnabla_nas.contrib as contrib
# from nnabla_nas.runner import Searcher, Trainer
from nnabla_nas.runner.search_pnas import Searcher
from nnabla_nas.runner import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension path. ex) cpu, cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='1',
                        help='Device ID the training run on. \
                        This is only valid if you specify `-c cudnn`.')
    parser.add_argument('--config-file', '-f', type=str, help='config file')
    args = parser.parse_args()

    config = json.load(open(args.config_file))
    config.update(vars(args))

    # setup context for nnabla
    ctx = get_extension_context(
        config['context'], device_id=config['device_id'])
    nn.set_default_context(ctx)

    # build the model
    attrs = config['network'].copy()
    algo = contrib.__dict__[attrs.pop("name")]
    if config['search']:
        Searcher(algo.SearchNet(**attrs), config).run()
    else:
        Trainer(algo.TrainNet(**attrs), config).run()
