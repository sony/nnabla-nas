import json
import os
import time
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.logger import logger
from scipy.special import softmax
from tensorboardX import SummaryWriter
from tqdm import tqdm

from .dataset.transformer import Compose, Normalizer


class ProgressMeter(object):
    def __init__(self, num_batches, meters, tb_writer=None, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = OrderedDict()
        for m in meters:
            self.meters[m.name] = m
        self.prefix = prefix
        self.writer = tb_writer

    def display(self, batch, key=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        key = key or [m.name for m in self.meters.values()]
        entries += [str(meter)
                    for meter in self.meters.values() if meter.name in key]
        print('\t'.join(entries))

    def __getitem__(self, key):
        return self.meters[key]

    def write(self, n_iter):
        if self.writer is not None:
            for m in self.meters.values():
                self.writer.add_scalar(m.name, m.avg, n_iter)

    def close(self):
        self.writer.close()

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


def get_standard_monitor(one_epoch, path):
    return ProgressMeter(
        num_batches=one_epoch,
        meters=[
            AverageMeter('train_loss', fmt=':5.3f'),
            AverageMeter('valid_loss', fmt=':5.3f'),
            AverageMeter('train_err', fmt=':5.3f'),
            AverageMeter('valid_err', fmt=':5.3f')
        ],
        tb_writer=SummaryWriter(
            os.path.join(path, 'tensorboard')
        )
    )


def sample(pvals, mode='sample'):
    """Return an index."""
    if mode == 'max':
        return np.argmax(pvals)
    return np.random.choice(len(pvals), p=pvals, replace=True)


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def dataset_transformer():
    normalize = Normalizer(
        mean=(0.49139968, 0.48215827, 0.44653124),
        std=(0.24703233, 0.24348505, 0.26158768),
        scale=255.0
    )
    train_transform = Compose([normalize])
    valid_transform = Compose([normalize])

    return train_transform, valid_transform


def _get_format(_alpha, _num_choices):
    offset = 0
    cell, prob = dict(), dict()
    for i in range(_num_choices):
        cell[i + 2], prob[i + 2] = list(), list()
        for j in range(i + 2):
            w = softmax(_alpha[j + offset].d.flatten())
            idx = int(np.argmax(w))
            if idx != 7:  # zero operation
                cell[i + 2].append([idx, j])
                prob[i + 2].append(float(w[idx]))
        offset += i + 2
    return cell, prob


def get_darts_arch(dart_model):
    # get current arch
    # Note: should be called right after training the architecture
    normal_cell, normal_prob = _get_format(
        dart_model._alpha_normal, dart_model._num_choices)
    reduce_cell, reduce_prob = _get_format(
        dart_model._alpha_reduce, dart_model._num_choices)

    normal_w, reduce_w = [], []

    for alpha in dart_model._alpha_normal:
        idx = np.argmax(alpha.d.flatten())
        normal_w.append(int(idx))

    for alpha in dart_model._alpha_reduce:
        idx = np.argmax(alpha.d.flatten())
        reduce_w.append(int(idx))

    return {
        'alpha_normal': normal_cell,
        'alpha_reduce': reduce_cell,
        'prob_normal': normal_prob,
        'prob_reduce': reduce_prob,
        'normal': normal_w,
        'reduce': reduce_w
    }


def drop_path(x, drop_prob):
    """Drop path function."""
    mask = F.rand(shape=(x.shape[0], 1, 1, 1))
    mask = F.greater_equal_scalar(mask, drop_prob)
    x = F.mul_scalar(x, 1.0 / (1 - drop_prob))
    x = F.mul2(x, mask)
    return x


def write_to_json_file(content, file_path):
    with open(file_path, 'w+') as file:
        json.dump(content, file,
                  ensure_ascii=False, indent=4,
                  default=lambda o: '<not serializable>')


def image_augmentation(image):
    out = F.random_crop(F.pad(image, (4, 4, 4, 4)), shape=(image.shape))
    return F.image_augmentation(out, flip_lr=True)
