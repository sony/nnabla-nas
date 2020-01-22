from .module import Module


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is None:
            modules = []
        self._modules = modules

    def add_module(self, module):
        self._modules.append(module)

    def __getitem__(self, idx):
        return self._modules[idx]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __call__(self, input, axis=0):
        return F.stack(*[m(input) for m in self._modules], axis=axis)

    def get_modules(self, prefix='', memo=None):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for i, module in enumerate(self._modules):
                name = prefix + '/_modules/{}'.format(i)
                yield from module.get_modules(name, memo)


class Sequential(ModuleList):
    def __init__(self, *args):
        super().__init__([])
        for module in args:
            self._modules.append(module)

    def __call__(self, input):
        out = input
        for module in self._modules:
            out = module(out)
        return out
