import json
import os
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.logger import logger
from scipy.special import softmax
from tensorboardX import SummaryWriter

from .dataset.transformer import Compose, Cutout, Normalizer
from nnabla.utils.profiler import GraphProfiler

class ProgressMeter(object):
    def __init__(self, num_batches, meters=[], path=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = OrderedDict()
        for m in meters:
            self.meters[m.name] = m
        self.writer = SummaryWriter(os.path.join(path, 'tensorboard'))

    def display(self, batch, key=None):
        entries = [self.batch_fmtstr.format(batch)]
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

    def update(self, tag, value, n):
        if tag not in self.meters:
            self.meters[tag] = AverageMeter(tag, fmt=':5.3f')
        self.meters[tag].update(value, n)

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
    for name, alpha in zip(['normal', 'reduce'],
                           [model._alpha_normal, model._alpha_reduce]):
        for k, v in zip(['alpha', 'prob', 'choice'],
                        parse_weights(alpha, model._num_choices)):
            memo[name + '_' + k] = v
    logger.info('Saving arch to {}'.format(file))
    write_to_json_file(memo, file)


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


def get_object_from_dict(module, args):
    if args is not None:
        class_name = args.pop('name')
        return module[class_name](**args)
    return None

class Estimator(object):
    """Estimator base class."""
    @property
    def memo(self):
        if '_memo' not in self.__dict__:
            self._memo = dict()
        return self._memo

    def get_estimation(self, module):
        """Returns the estimation of the whole module."""
        return sum(self.predict(m) for _, m in module.get_modules()
                   if len(m.modules) == 0 and m.need_grad)

    def reset(self):
        """Clear cache."""
        self.memo.clear()

    def predict(self, module):
        """Predicts the estimation for a module."""
        raise NotImplementedError


class MemoryEstimator(Estimator):
    """Estimator for the memory used."""

    def predict(self, module):
        idm = id(module)
        if idm not in self.memo:
            self.memo[idm] = sum(np.prod(p.shape)
                                 for p in module.parameters.values())
        return self.memo[idm]


class LatencyEstimator(Estimator):
    """Latency estimator.

    Args:
        device_id (int): gpu device id.
        ext_name (str): Extension name. e.g. 'cpu', 'cuda', 'cudnn' etc.
        n_run (int): This argument specifies how many times the each functions
            execution time are measured. Default value is 10.
    """

    def __init__(self, n_run=10):
        ctx = nn.context.get_current_context()
        self._device_id = int(ctx.device_id)
        self._ext_name = ctx.backend[0].split(':')[0]
        self._n_run = n_run

    def predict(self, module):
        idm = id(module)
        if idm not in self.memo:
            self.memo[idm] = dict()
        mem = self.memo[idm]
        key = '-'.join([str(k) for k in module.inputs])

        if key not in mem:
            state = module.training
            module.apply(training=False)  # turn off training

            try:
                # run profiler
                nnabla_vars = [nn.Variable(s) for s in module.input_shape]
                runner = GraphProfiler(module.call(*nnabla_vars),
                                    device_id=self._device_id,
                                    ext_name=self._ext_name,
                                    n_run=self._n_run)
                runner.time_profiling_forward()
                mem[key] = float(runner.result['forward'][0].mean_time)
            except:
                mem[key] = float(0.0)
                print("module {} cannot be profiled. Using latency 0.0 instead".format(key))
            module.apply(training=state)  # recover training state
        return mem[key]

