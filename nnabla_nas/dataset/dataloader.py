class DataLoader(object):

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
