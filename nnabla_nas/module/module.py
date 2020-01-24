from collections import OrderedDict

import nnabla as nn

from .parameter import Parameter


class Module(object):
    r"""Module base for all nnabla neural network modules."""

    @property
    def modules(self):
        r"""Return an `OrderedDict` containing immediate modules."""
        if '_modules' not in self.__dict__:
            self.__dict__['_modules'] = OrderedDict()
        return self._modules

    @property
    def parameters(self):
        r"""Return an `OrderedDict` containing immediate parameters."""
        if '_parameters' not in self.__dict__:
            self.__dict__['_parameters'] = OrderedDict()
        return self._parameters

    @property
    def training(self):
        r"""The training mode of module."""
        if '_training' not in self.__dict__:
            self.__dict__['_training'] = True
        return self._training

    @training.setter
    def training(self, mode):
        setattr(self, '_training', mode)
        for module in self.modules.values():
            module.training = mode

    @property
    def need_grad(self):
        r"""Whether the module needs gradient."""
        if '_need_grad' not in self.__dict__:
            self.__dict__['_need_grad'] = True
        return self._need_grad

    @need_grad.setter
    def need_grad(self, mode):
        setattr(self, '_need_grad', mode)
        for module in self.modules.values():
            module.need_grad = mode

    @property
    def inputs(self):
        r"""Return a list of inputs used during `call` function."""
        if '_inputs' not in self.__dict__:
            self.__dict__['_inputs'] = list()
        return self._inputs

    @inputs.setter
    def inputs(self, v):
        setattr(self, '_inputs', v)

    @property
    def outputs(self):
        r"""Return a list of outputs used during `call` function."""
        if '_outputs' not in self.__dict__:
            self.__dict__['outputs'] = list()
        return self._outputs

    @outputs.setter
    def outputs(self, v):
        setattr(self, '_outputs', v)

    def __getattr__(self, name):
        if name in self.modules:
            return self.modules[name]
        if name in self.parameters:
            return self.parameters[name]
        return object.__getattr__(self, name)

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                d.pop(name, None)
        remove_from(self.__dict__, self.modules, self.parameters)
        if isinstance(value, Module):
            self.modules[name] = value
        elif isinstance(value, Parameter):
            self.parameters[name] = value
        else:  # avoid conflict with property
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self.parameters:
            del self.parameters[name]
        elif name in self.modules:
            del self.modules[name]
        else:
            object.__delattr__(self, name)

    def apply(self, **kargs):
        r"""Helper for setting property, then return self."""
        for key, value in kargs.items():
            setattr(self, key, value)
        return self

    def get_modules(self, prefix='', memo=None):
        r"""Return an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself."""
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self.modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('/' if prefix else '') + name
                for m in module.get_modules(submodule_prefix, memo):
                    yield m

    def get_parameters(self, grad_only=False):
        r"""Return an `OrderedDict` containing all parameters in the module."""
        params = OrderedDict()
        for prefix, module in self.get_modules():
            if grad_only and not module.need_grad:
                continue
            for name, p in module.parameters.items():
                if grad_only and not p.need_grad:
                    continue
                key = prefix + ('/' if prefix else '') + name
                params[key] = p
        return params

    def set_parameters(self, params, raise_if_missing=False):
        r"""Set parameters for the module."""
        for prefix, module in self.get_modules():
            for name, p in module.parameters.items():
                key = prefix + ('/' if prefix else '') + name
                if key in params:
                    p.d = params[key].d.copy()
                    nn.logger.info(f'`{key}` loaded.)')
                elif raise_if_missing:
                    raise ValueError(
                        f'A child module {name} cannot be found in '
                        '{this}. This error is raised because '
                        '`raise_if_missing` is specified '
                        'as True. Please turn off if you allow it.')
        return self

    def __key_format__(self, key):
        r"""Set the submodule representation."""
        return f'.{key}'

    def __extra_repr__(self):
        r"""Set the extra representation for the module."""
        return ''

    def __repr__(self):
        r"""Return str representtation of the module."""
        main_str = f'{self.__class__.__name__}(' + self.__extra_repr__()
        sub_str = ''
        for key, module in self.modules.items():
            m_repr = repr(module).split('\n')
            head = [self.__key_format__(key) + ': ' + m_repr.pop(0)]
            tail = [m_repr.pop()] if len(m_repr) else []
            m_repr = [' '*2 + line for line in (head + m_repr + tail)]
            sub_str += '\n' + '\n'.join(m_repr)
        main_str += sub_str + ('\n' if sub_str else '') + ')'
        return main_str

    def __call__(self, *args, **kargs):
        self.inputs = args
        self.outputs = self.call(*args, **kargs)
        return self.outputs

    def call(self, *args, **kargs):
        r"""Implement the call of module. Inmediate inputs should only be
        Variables."""
        raise NotImplementedError
