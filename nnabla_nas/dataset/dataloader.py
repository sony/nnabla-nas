class DataLoader(object):
    r""" Dataloader class.

    Combines a data iterator and a transform (on numpy), and provides an
    iterable over the given dataset.  All subclasses should overwrite
    `__len__` and `next`.

    Args:
        data_iterator (iterator): Data iterator
        transform (object, optional): A transform to be applied on a sample.
            Defaults to None.
    """

    def __init__(self, data_iterator, transform=None):
        self.data_iterator = data_iterator
        self.transform = transform

    def next(self):
        x, t = self.data_iterator.next()
        if self.transform:
            x = self.transform(x)
        return x, t

    def __len__(self):
        return self.data_iterator.size
