import json
import os
import sys
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.logger import logger
from scipy.special import softmax
from tensorboardX import SummaryWriter

from .dataset.transformer import Compose
from .dataset.transformer import Cutout
from .dataset.transformer import Normalizer
from .visualization import visualize


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
        """Shows a message.

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
        """Displays current values for meters.

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
        self.tb.close()
        self.file.close()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def reset(self):
        for m in self.meters.values():
            m.reset()


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
    """Return an index."""
    if mode == 'max':
        return np.argmax(pvals)
    return np.random.choice(len(pvals), p=pvals, replace=False)


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def dataset_transformer(conf):
    normalize = Normalizer(
        mean=(0.49139968, 0.48215827, 0.44653124),
        std=(0.24703233, 0.24348505, 0.26158768),
        scale=255.0
    )
    train_transform = Compose([normalize])
    if conf.get('cutout', 0) > 0:
        train_transform.append(Cutout(conf['cutout']))
    valid_transform = Compose([normalize])

    return train_transform, valid_transform


def parse_weights(alpha, num_choices):
    offset = 0
    cell, prob, choice = dict(), dict(), dict()
    for i in range(num_choices):
        cell[i + 2], prob[i + 2] = list(), list()
        W = [softmax(alpha[j + offset].d.flatten()) for j in range(i + 2)]
        # Note: Zero Op shouldn't be included
        edges = sorted(range(i + 2), key=lambda k: -max(W[k][:-1]))
        for j, k in enumerate(edges):
            if j < 2:  # select the first two best Ops
                idx = np.argmax(W[k][:-1])
                cell[i + 2].append([int(idx), k])
                prob[i + 2].append(float(W[k][idx]))
                choice[k + offset] = int(idx)
            else:  # assign Zero Op to the rest
                choice[k + offset] = int(len(W[k]) - 1)
        offset += i + 2
    return cell, prob, choice


def save_dart_arch(model, output_path):
    memo = dict()
    for name, alpha in zip(['normal', 'reduce'],
                           [model._alpha[0], model._alpha[1]]):
        for k, v in zip(['alpha', 'prob', 'choice'],
                        parse_weights(alpha, model._num_choices)):
            memo[name + '_' + k] = v
    arch_file = output_path + 'arch.json'
    logger.info('Saving arch to {}'.format(arch_file))
    write_to_json_file(memo, arch_file)
    visualize(arch_file, output_path)


def drop_path(x):
    """Drop path function. Taken from Yashima code"""
    drop_prob = nn.parameter.get_parameter_or_create(
        "drop_rate",
        shape=(1, 1, 1, 1),
        need_grad=False
    )
    mask = F.rand(shape=(x.shape[0], 1, 1, 1))
    mask = F.greater_equal(mask, drop_prob)
    x = F.div2(x, 1 - drop_prob)
    x = F.mul2(x, mask)
    return x


def load_parameters(path):
    """Loads the parameters from a file.

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


def get_params_size(params):
    return np.sum(np.prod(p.shape) for p in params.values())


def get_object_from_dict(module, args):
    if args is not None:
        args = args.copy()
        class_name = args.pop('name')
        return module[class_name](**args)
    return None
