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

import os
import sys

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.logger import logger
from omegaconf import DictConfig, OmegaConf
from ..helper import get_output_path

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
    print(" Please make sure that your installed nnabla-ext-cuda is appropriate for your environment.")
    raise

from nnabla_nas.utils.cli.args import Configuration
from nnabla_nas import contrib
from nnabla_nas import runner
from nnabla_nas.utils.helper import CommunicatorWrapper


def main(cfg: DictConfig):

    config = OmegaConf.to_object(cfg)
    args_ = config['args']

    # setup cuda visible
    if args_['device_id'] != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args_['device_id']

    # setup context for nnabla
    ctx = get_extension_context(
        args_['context'],
        device_id='0',
        type_config=args_['type_config']
    )

    # setup for distributed training. Needs to go into hyperparameters
    comm = CommunicatorWrapper(ctx)
    args_['comm'] = comm
    args_['event'] = StreamEventHandler(int(comm.ctx.device_id))

    nn.set_default_context(comm.ctx)

    if comm.n_procs > 1 and comm.rank == 0:
        n_procs = comm.n_procs
        logger.info(f'Distributed training with {n_procs} processes.')

    # build the model
    name, attributes = list(config['network'].items())[0]
    algorithm = contrib.__dict__[name]
    model = algorithm.SearchNet(**attributes) if args_['search'] else \
        algorithm.TrainNet(**attributes)

    # Get all arguments for the runner
    objects = Configuration(config)

    # Logging the output path of the experiment
    output_path = get_output_path()
    logger.info("Saving experiment results to %s" % output_path)

    runner.__dict__[args_['algorithm']](
        model,
        optimizer=objects.optimizer,
        regularizer=objects.regularizer,
        dataloader=objects.dataloader,
        hparams=config['hparams'],  # hyperparameters
        args=args_  # other parameters needed when training
    ).run()
