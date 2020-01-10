from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F

from .parameter import Parameter


class Module(object):

    def __init__(self):
        self.need_grad = True
        self.training = True

    def update_grad(self, mode=True, memo=None):
        """Update need_grad."""
        if memo is None:
            memo = set()

        if self.need_grad != mode and self not in memo:
            memo.add(self)
            # update grads for the current module
            self.need_grad = mode
            for v in self.__dict__.values():
                if isinstance(v, Parameter):
                    v.need_grad = mode
            # update its children
            for module in self.children():
                module.update_grad(mode, memo)

        return self

    def train(self, mode=True, memo=None):
        r"""Sets the module in training mode.
        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training or
        evaluation mode, if they are affected, e.g. :class:`Dropout`,
        :class:`BatchNormalization`,
        etc.
        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.
        Returns:
            Module: self
        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            self.training = mode
            for module in self.children():
                module.train(mode, memo)
        return self

    def eval(self):
        r"""Sets the module in evaluation mode.
        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training or
        evaluation mode, if they are affected, e.g. :class:`Dropout`,
        :class:`BatchNorm`, etc.
        This is equivalent with :meth:`self.train(False)`.
        Returns:
            Module: self
        """
        return self.train(False)

    def children(self):
        r"""Returns an iterator over immediate children modules.
        Yields:
            Module: a child module
        """
        for module in self.__dict__.values():
            if isinstance(module, Module):
                yield module

    def get_modules(self, prefix='', memo=None):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield (prefix, self)
            for name, module in self.__dict__.items():
                if not isinstance(module, Module) or module in memo:
                    continue
                sub_prefix = '{}/{}'.format(prefix,
                                            name) if prefix != '' else name
                for m in module.get_modules(sub_prefix, memo):
                    yield m

    def get_parameters(self, grad_only=False):
        params = OrderedDict()
        for (prefix, module) in self.get_modules():
            if grad_only and not module.need_grad:
                continue
            for k, v in module.__dict__.items():
                if not isinstance(v, Parameter):
                    continue
                name = '{}/{}'.format(prefix, k)
                if grad_only and not v.need_grad:
                    continue
                params[name] = v
        return params

    def save_parameters(self, path, params=None, grad_only=False):
        """Save all parameters into a file with the specified format.
        Currently hdf5 and protobuf formats are supported.
        Args:
            path : path or file object
            grad_only (bool, optional): Return parameters with `need_grad`
                option as `True`.
        """
        params = params or self.get_parameters(grad_only=grad_only)
        nn.save_parameters(path, params)

    def load_parameters(self, path):
        """Load parameters from a file with the specified format.
        Args:
            path : path or file object
        """
        nn.load_parameters(path)
        for v in self.get_modules():
            if not isinstance(v, tuple):
                continue
            prefix, module = v
            for k, v in module.__dict__.items():
                if not isinstance(v, Parameter):
                    continue
                pname = k
                name = "{}/{}".format(prefix, pname)
                # Substitute
                param0 = v
                param1 = nn.parameter.pop_parameter(name)
                if param0 is None:
                    raise ValueError(
                        "Model does not have {} parameter.".format(name))
                if param1:
                    param0.d = param1.d.copy()
                    nn.logger.info("`{}` loaded.)".format(name))

    def __call__(self, *input):
        raise NotImplementedError


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
