import operator
from collections import OrderedDict

from .module import Module


def _get_abs_string_index(obj, idx):
    """Get the absolute index for the list of modules"""
    idx = operator.index(idx)
    if not (-len(obj) <= idx < len(obj)):
        raise IndexError('index {} is out of range'.format(idx))
    if idx < 0:
        idx += len(obj)
    return str(idx)


class ModuleList(Module):
    r"""Hold submodules in a list. This implementation mainly follows
    the Pytorch implementation."""

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self += modules

    def append(self, module):
        r"""Appends a given module to the end of the list."""
        setattr(self, str(len(self)), module)
        return self

    def extend(self, modules):
        for module in modules:
            self.append(module)
        return self

    def insert(self, index, module):
        r"""Insert a given module before a given index in the list."""
        for i in range(len(self), index, -1):
            self.modules[str(i)] = self.modules[str(i - 1)]
        self.modules[str(index)] = module
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(list(self.modules.values())[index])
        index = _get_abs_string_index(self, index)
        return self.modules[index]

    def __setitem__(self, index, module):
        index = _get_abs_string_index(self, index)
        self.modules[str(index)] = module

    def __delitem__(self, index):
        if isinstance(index, slice):
            for k in range(len(self.modules))[index]:
                delattr(self, str(k))
        else:
            delattr(self, _get_abs_string_index(self, index))
        indices = [str(i) for i in range(len(self.modules))]
        self._modules = OrderedDict(list(zip(indices, self.modules.values())))

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def __key_format__(self, key):
        return f'[{key}]'


class Sequential(ModuleList):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ordered dict of modules can also be
    passed in.
    """

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                setattr(self, key, module)
        else:
            for idx, module in enumerate(args):
                setattr(self, str(idx), module)

    def call(self, input):
        for module in self.modules.values():
            input = module(input)
        return input
