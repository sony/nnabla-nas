import json
import os
from collections import OrderedDict

import nnabla.functions as F
import numpy as np
from nnabla.logger import logger
from scipy.special import softmax
from tensorboardX import SummaryWriter

from .dataset.transformer import Compose, Cutout, Normalizer


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

    def write_image(self, tag, image_tensor, n_iter):
        self.writer.add_image(tag, image_tensor, n_iter)

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


def dataset_transformer(conf):
    normalize = Normalizer(
        mean=(0.49139968, 0.48215827, 0.44653124),
        std=(0.24703233, 0.24348505, 0.26158768),
        scale=255.0
    )
    train_transform = Compose([normalize])
    if 'cutout' in conf and conf['cutout']:
        train_transform.append(Cutout(conf['cutout_length']))
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


def save_dart_arch(model, file):
    memo = dict()
    for name, alpha in zip(['normal', 'reduce'], [model._alpha_normal, model._alpha_reduce]):
        for k, v in zip(['alpha', 'prob', 'choice'], parse_weights(alpha, model._num_choices)):
            memo[name + '_' + k] = v
    logger.info('Saving arch to {}'.format(file))
    write_to_json_file(memo, file)


def drop_path(x, drop_prob):
    """Drop path function."""
    mask = F.rand(shape=(x.shape[0], 1, 1, 1))
    mask = F.greater_equal(mask, drop_prob)
    x = F.div2(x, 1 - drop_prob)
    x = F.mul2(x, mask)
    return x


def write_to_json_file(content, file_path):
    with open(file_path, 'w+') as file:
        json.dump(content, file,
                  ensure_ascii=False, indent=4,
                  default=lambda o: '<not serializable>')


def image_augmentation(image):
    out = F.random_crop(F.pad(image, (4, 4, 4, 4)), shape=(image.shape))
    out = F.image_augmentation(out, flip_lr=True)
    out.need_grad = False
    return out


def get_params_size(params):
    return np.sum(np.prod(p.shape) for p in params.values())
