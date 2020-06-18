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

import nnabla as nn
import nnabla.functions as F
import numpy as np


class Normalize(object):
    r"""Normalizes a input image with mean and standard deviation.

    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input image i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        scale (float): Scales the inputs by a scalar.
    """

    def __init__(self, mean, std, scale):
        self._mean = nn.Variable.from_numpy_array(
            np.reshape(mean, (1, 3, 1, 1)))
        self._std = nn.Variable.from_numpy_array(
            np.reshape(std, (1, 3, 1, 1)))
        self._scale = scale

    def __call__(self, input):
        out = F.mul_scalar(input, self._scale)
        out = F.sub2(out, self._mean)
        out = F.div2(out, self._std)
        return out

    def __str__(self):
        return self.__class__.__name__
        + f'(mean={self._mean.d}, std={self._std.d}, scale={self._scale})'


class Compose(object):
    r"""Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to
            compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def append(self, transform):
        r"""Appends a transfomer to the end.

        Args:
            transform (Transformer): The transforme to append.
        """
        self.transforms.append(transform)

    def __str__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(str(t))
        format_string += '\n)'
        return format_string


class Cutout(object):
    r"""Cutout layer.

    Cutout is a simple regularization technique for convolutional neural
    networks that involves removing contiguous sections of input images,
    effectively augmenting the dataset with partially occluded versions of
    existing samples.

    Args:
        length (int): The lenth of region, which will be cutout.
        prob (float, optional): Probability of earasing. Defaults to 0.5.

    References:
        [1] DeVries, Terrance, and Graham W. Taylor. "Improved regularization
                of convolutional neural networks with cutout." arXiv preprint
                arXiv:1708.04552 (2017).
    """

    def __init__(self, length, prob=0.5, seed=-1):
        self._length = length
        self._prob = prob
        self._seed = seed

    def __call__(self, images):
        ratio = self._length**2 / np.prod(images.shape[2:])
        area_ratios = (ratio, ratio)
        aspect_ratios = (1.0, 1.0)
        out = F.random_erase(images,
                             seed=self._seed,
                             prob=self._prob,
                             replacements=(0.0, 0.0),
                             aspect_ratios=aspect_ratios,
                             area_ratios=area_ratios)
        out = out.apply(need_grad=False)
        return out

    def __str__(self):
        return self.__class__.__name__ + f'(length={self._length})'


class Resize(object):
    r"""Resize an ND array with interpolation.

    Args:
        size (tuple of `int`): The output sizes for axes. If this is
            given, the scale factors are determined by the output sizes and the
            input sizes.
        interpolation (str): Interpolation mode chosen from
            ('linear'|'nearest'). The default is 'linear'.
    """

    def __init__(self, size, interpolation='linear'):
        self._size = size
        self._interpolation = interpolation

    def __call__(self, input):
        out = F.interpolate(input, output_size=self._size,
                            mode=self._interpolation)
        return out

    def __str__(self):
        return self.__class__.__name__ + (
            f'(size={self._size}, '
            f'interpolation={self._interpolation})'
        )


class RandomCrop(object):
    r"""RandomCrop randomly extracts a portion of an array.

    Args:
        shape ([type]): [description]
        pad_width (tuple of `int`, optional): Iterable of *before* and *after*
            pad values. Defaults to None. Pad the input N-D array `x` over the
            number of dimensions given by half the length of the `pad_width`
            iterable, where every two values in `pad_width` determine the
            before and after pad size of an axis. The `pad_width` iterable
            must hold an even number of positive values which may cover all or
            fewer dimensions of the input variable `x`.
    """

    def __init__(self, shape, pad_width=None):
        self._shape = shape
        self._pad_width = pad_width

    def __call__(self, input):
        if self._pad_width is not None:
            input = F.pad(input, self._pad_width)
        return F.random_crop(input, shape=self._shape)

    def __str__(self):
        return self.__class__.__name__ + (
            f'(shape={self._shape}, '
            f'pad_width={self._pad_width})'
        )


class RandomHorizontalFlip(object):
    r"""Horizontally flip the given Image randomly with a probability 0.5."""

    def __call__(self, input):
        return F.image_augmentation(input, flip_lr=True)

    def __str__(self):
        return self.__class__.__name__ + '(p=0.5)'


class RandomVerticalFlip(object):
    r"""Vertically flip the given PIL Image randomly with a probability 0.5."""

    def __call__(self, input):
        return F.image_augmentation(input, flip_ud=True)

    def __str__(self):
        return self.__class__.__name__ + '(p=0.5)'


class RandomRotation(object):
    def __call__(self, input):
        raise NotImplementedError


class CenterCrop(object):
    def __call__(self, input):
        raise NotImplementedError


class Lambda(object):
    r"""Apply a user-defined lambda as a transform.

    Args:
        func (function): Lambda/function to be used for transform.
    """

    def __init__(self, func):
        assert callable(func), repr(type(func).__name__)
        + " object is not callable"
        self._func = func

    def __call__(self, input):
        return self._func(input)

    def __str__(self):
        return self.__class__.__name__ + '()'
