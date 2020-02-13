import json
import os
import sys
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
import numpy as np
from tensorboardX import SummaryWriter

from .dataset.transformer import Compose
from .dataset.transformer import Cutout
from .dataset.transformer import Normalize


class ProgressMeter(object):
    r"""A Progress Meter.

        Args:
            num_batches (int): The number of batches per epoch.
            path (str, optional): Path to save tensorboard and log file.
                Defaults to None.
    """

    def __init__(self, num_batches, path=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = OrderedDict()
        self.terminal = sys.stdout
        self.tb = SummaryWriter(os.path.join(path, 'tensorboard'))
        self.file = open(os.path.join(path, 'log.txt'), 'w')

    def info(self, message, view=True):
        r"""Shows a message.

        Args:
            message (str): The message.
            view (bool, optional): If shows to terminal. Defaults to True.
        """
        if view:
            self.terminal.write(message)
            self.terminal.flush()
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


def sample(pvals, mode='sample'):
    r"""Returns random int according the sampling `mode` (e.g., `max`, `full`,
        or `sample`).

    Args:
        pvals (np.array): The probability values.
        mode (str, optional): The sampling `mode`. Defaults to 'sample'.

    Returns:
        [type]: [description]
    """
    if mode == 'max':
        return np.argmax(pvals)
    return np.random.choice(len(pvals), p=pvals, replace=False)


def dataset_transformer(conf):
    r"""Returns data transformers for training and validating the model.

    Args:
        conf (dict): A dictionary containning configurations.

    Returns:
        (Transformer, Transformer): Training and validating transformers.
    """
    normalize = Normalize(
        mean=(0.49139968, 0.48215827, 0.44653124),
        std=(0.24703233, 0.24348505, 0.26158768),
        scale=1./255.0
    )
    train_transform = Compose([normalize])
    if conf.get('cutout', 0) > 0:
        train_transform.append(Cutout(conf['cutout']))
    valid_transform = Compose([normalize])

    return train_transform, valid_transform


def load_parameters(path):
    r"""Loads the parameters from a file.

    Args:
        path (str): The path to file.

    Returns:
        OrderedDict: An `OrderedDict` containing parameters.
    """
    with nn.parameter_scope('', OrderedDict()):
        nn.load_parameters(path)
        params = nn.get_parameters()
    return params


def write_to_json_file(content, file_path):
    r"""Saves a dictionary to a json file.

    Args:
        content (dict): The content to save.
        file_path (str): The file path.
    """
    with open(file_path, 'w+') as file:
        json.dump(content, file,
                  ensure_ascii=False, indent=4,
                  default=lambda o: '<not serializable>')


def data_augment(image):
    r"""Performs standard data augmentations.

    Args:
        image (numpy.array): The input image.

    Returns:
        numpy.array: The output.
    """
    out = F.random_crop(F.pad(image, (4, 4, 4, 4)), shape=(image.shape))
    out = F.image_augmentation(out, flip_lr=True)
    out.need_grad = False
    return out


def count_parameters(params):
    r"""Counts the number of parameters.

    Args:
        params (OrderedDict): The dictionary containing parameters.

    Returns:
        int: The total number of parameters.
    """
    return np.sum(np.prod(p.shape) for p in params.values())
