import numpy as np


class Normalizer(object):

    def __init__(self, mean, std, scale):
        self.mean = np.reshape(mean, (1, 3, 1, 1))
        self.std = np.reshape(std, (1, 3, 1, 1))
        self.scale = scale

    def __call__(self, x):
        out = x.copy() / self.scale
        out = (out - self.mean) / self.std
        return out

    def __repr__(self):
        return self.__class__.__name__
        + '(mean={0}, std={1})'.format(self.mean.flatten(),
                                       self.std.flatten())


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
