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

from collections import OrderedDict
import json
import os
import sys

from nnabla import random
import nnabla.communicators as C
from nnabla.ext_utils import get_extension_context
import numpy as np

from .tensorboard import SummaryWriter


class ProgressMeter(object):
    r"""A Progress Meter.

        Args:
            num_batches (int): The number of batches per epoch.
            path (str, optional): Path to save tensorboard and log file.
                Defaults to None.
    """

    def __init__(self, num_batches, path=None, quiet=False):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = OrderedDict()
        self.terminal = sys.stdout
        self.quiet = quiet
        if not self.quiet:
            self.tb = SummaryWriter(os.path.join(path, 'tensorboard'))
            self.file = open(os.path.join(path, 'log.txt'), 'w')

    def info(self, message, view=True):
        r"""Shows a message.

        Args:
            message (str): The message.
            view (bool, optional): If shows to terminal. Defaults to True.
        """
        if view and not self.quiet:
            self.terminal.write(message)
            self.terminal.flush()
        if not self.quiet:
            self.file.write(message)
            self.file.flush()

    def display(self, batch, key=None):
        r"""Displays current values for meters.

        Args:
            batch (int): The number of batch.
            key ([type], optional): [description]. Defaults to None.
        """

        entries = [self.batch_fmtstr.format(batch)]
        key = key or [m.name for m in self.meters.values()]
        entries += [str(meter) for meter in self.meters.values()
                    if meter.name in key]
        self.info('\t'.join(entries) + '\n')

    def __getitem__(self, key):
        return self.meters[key]

    def write(self, n_iter):
        r"""Writes info to tensorboard

        Args:
            n_iter (int): The n-th iteration.
        """
        if self.quiet:
            return

        for m in self.meters.values():
            self.tb.add_scalar(m.name, m.avg, n_iter)

    def update(self, tag, value, n=1):
        r"""Updates the meter.

        Args:
            tag (str): The tag name.
            value (number): The value to update.
            n (int, optional): The len of minibatch. Defaults to 1.
        """
        if tag not in self.meters:
            self.meters[tag] = AverageMeter(tag, fmt=':5.3f')
        self.meters[tag].update(value, n)

    def close(self):
        r"""Closes all the file descriptors."""
        if not self.quiet:
            self.tb.close()
            self.file.close()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def reset(self):
        r"""Resets the ProgressMeter."""
        for m in self.meters.values():
            m.reset()


class AverageMeter(object):
    r"""Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def sample(pvals, mode='sample', rng=None):
    r"""Returns random int according the sampling `mode` (e.g., `max`, `full`,
        or `sample`).

    Args:
        pvals (np.array): The probability values.
        mode (str, optional): The sampling `mode`. Defaults to 'sample'.
        rng (numpy.random.RandomState): Random generator for random choice.

    Returns:
        [type]: [description]
    """
    if mode == 'max':
        return np.argmax(pvals)
    if rng is None:
        rng = random.prng
    return rng.choice(len(pvals), p=pvals, replace=False)


def write_to_json_file(content, file_path):
    r"""Saves a dictionary to a json file.

    Args:
        content (dict): The content to save.
        file_path (str): The file path.
    """
    with open(file_path, 'w+') as file:
        json.dump(content, file,
                  ensure_ascii=False, indent=4,
                  default=lambda o: o.__class__.__name__)


def count_parameters(params):
    r"""Counts the number of parameters.

    Args:
        params (OrderedDict): The dictionary containing parameters.

    Returns:
        int: The total number of parameters.
    """
    return np.sum(np.prod(p.shape) for p in params.values())


def create_float_context(ctx):
    ctx_float = get_extension_context(ctx.backend[0].split(':')[
                                      0], device_id=ctx.device_id)
    return ctx_float


class CommunicatorWrapper(object):
    def __init__(self, ctx):
        try:
            comm = C.MultiProcessDataParallelCommunicator(ctx)
        except Exception as e:
            print(e)
            print(('No communicator found. Running with a single process. '
                   'If you run this with MPI processes, all processes will '
                   'perform totally same.'))
            self.n_procs = 1
            self.rank = 0
            self.ctx = ctx
            self.ctx_float = create_float_context(ctx)
            self.comm = None
            return

        comm.init()
        self.n_procs = comm.size
        self.rank = comm.rank
        self.ctx = ctx
        self.ctx.device_id = str(self.rank)
        self.ctx_float = create_float_context(self.ctx)
        self.comm = comm

    def all_reduce(self, params, division, inplace):
        if self.n_procs == 1:
            # skip all reduce since no processes have to be all-reduced
            return
        self.comm.all_reduce(params, division=division, inplace=inplace)
