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
import json
import os
import sys

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.logger import logger

try:
    from nnabla_ext.cuda import StreamEventHandler
except ModuleNotFoundError:
    print("ERROR: You need to install NNabla CUDA extension by yourself.")
    print("Select one of following that suits your environment.")
    print("")
    print("   pip install nnabla-ext-cuda90")
    print("   pip install nnabla-ext-cuda100")
    print("   pip install nnabla-ext-cuda101")
    print("   pip install nnabla-ext-cuda102 (Comming soon)")
    sys.exit(-1)
except ImportError:
    print("ERROR: CUDA extension installed but could not initialized.")
    print(" Please make sure that your installed nnabla-ext-cuda??? appropriate for your environment.")
    raise

from nnabla_nas.utils.cli.args import Configuration
from nnabla_nas import contrib
from nnabla_nas import runner
from nnabla_nas.utils.helper import CommunicatorWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension module. 'cudnn' is highly recommended.")
    parser.add_argument("--device-id", "-d", type=str, default='-1',
                        help='A list of device ids to use, e.g., `0,1,2,3`.\
                        This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type configuration.')
    parser.add_argument('--search', '-s', action='store_true',
                        help='Whether it is searching for the architecture.')
    parser.add_argument('--algorithm', '-a', type=str, default='DartsSeacher',
                        choices=runner.__all__, help='Which algorithm to use.')
    parser.add_argument('--config-file', '-f', type=str,
                        help='The configuration file for the experiment.')
    parser.add_argument('--output-path', '-o', type=str, help='Path to save the monitoring log files.')

    options = parser.parse_args()

    config = json.load(open(options.config_file)) if options.config_file else dict()
    hparams = config['hparams']

    hparams.update({k: v for k, v in vars(options).items() if v is not None})

    # setup cuda visible
    if hparams['device_id'] != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = hparams['device_id']

    # setup context for nnabla
    ctx = get_extension_context(
        hparams['context'],
        device_id='0',
        type_config=hparams['type_config']
    )

    # setup for distributed training
    hparams['comm'] = CommunicatorWrapper(ctx)
    hparams['event'] = StreamEventHandler(int(hparams['comm'].ctx.device_id))

    nn.set_default_context(hparams['comm'].ctx)

    if hparams['comm'].n_procs > 1 and hparams['comm'].rank == 0:
        n_procs = hparams['comm'].n_procs
        logger.info(f'Distributed training with {n_procs} processes.')

    # build the model
    name, attributes = list(config['network'].items())[0]
    algorithm = contrib.__dict__[name]
    model = algorithm.SearchNet(**attributes) if hparams['search'] else \
        algorithm.TrainNet(**attributes)

    # Get all arguments for the runner
    conf = Configuration(config)

    runner.__dict__[hparams['algorithm']](
        model,
        optimizer=conf.optimizer,
        dataloader=conf.dataloader,
        args=conf.hparams
    ).run()
