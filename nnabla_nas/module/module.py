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

from collections import OrderedDict

import nnabla as nn

from .parameter import Parameter


class Module(object):
    r"""Module base for all nnabla neural network modules.

    Your models should also subclass this class. Modules can also contain
    other Modules, allowing to nest them in a tree structure.
    """

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
        self.__dict__['_training'] = mode

    @property
    def need_grad(self):
        r"""Whether the module needs gradient."""
        if '_need_grad' not in self.__dict__:
            self.__dict__['_need_grad'] = True
        return self._need_grad

    @need_grad.setter
    def need_grad(self, mode):
        self.__dict__['_need_grad'] = mode

    @property
    def input_shapes(self):
        r"""Return a list of input shapes used during `call` function."""
        if '_input_shapes' not in self.__dict__:
            self.__dict__['_input_shapes'] = list()
        return self._input_shapes

    @input_shapes.setter
    def input_shapes(self, v):
        setattr(self, '_input_shapes', v)

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
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self.parameters:
            del self.parameters[name]
        elif name in self.modules:
            del self.modules[name]
        else:
            object.__delattr__(self, name)

    def apply(self, memo=None, **kargs):
        r"""Helper for setting property recursively, then returns self."""
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            for key, value in kargs.items():
                setattr(self, key, value)
            for module in self.modules.values():
                module.apply(memo, **kargs)
        return self

    def get_modules(self, prefix='', memo=None):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            prefix (str, optional): Additional prefix to name modules.
                Defaults to ''.
            memo (dict, optional): Memorize all parsed modules.
                Defaults to None.

        Yields:
            (str, Module): a submodule.
        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self.modules.items():
                submodule_prefix = prefix + ('/' if prefix else '') + name
                for m in module.get_modules(submodule_prefix, memo):
                    yield m

    def get_parameters(self, grad_only=False):
        r"""Return an `OrderedDict` containing all parameters in the module.

        Args:
            grad_only (bool, optional): If need_grad=True is required.
                Defaults to False.

        Returns:
            OrderedDict: A dictionary containing parameters of module.
        """
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
        r"""Set parameters for the module.

        Args:
            params (OrderedDict): The parameters which will be loaded.
            raise_if_missing (bool, optional): Raise exception if some
                parameters are missing. Defaults to `False`.

        Raises:
            ValueError: Parameters are not found.
        """
        for prefix, module in self.get_modules():
            for name, p in module.parameters.items():
                key = prefix + ('/' if prefix else '') + name
                if key in params:
                    p.d = params[key].d.copy()
                    nn.logger.info(f'`{key}` loaded.')
                elif raise_if_missing:
                    raise ValueError(
                        f'A child module {name} cannot be found in '
                        '{this}. This error is raised because '
                        '`raise_if_missing` is specified '
                        'as True. Please turn off if you allow it.')

    def save_parameters(self, path, params=None, grad_only=False):
        r"""Saves the parameters to a file.

        Args:
            path (str): Path to file.
            params (OrderedDict, optional): An `OrderedDict` containing
                parameters. If params is `None`, then the current parameters
                will be saved.
            grad_only (bool, optional): If need_grad=True is required for
                parameters which will be saved. Defaults to False.
        """
        params = params or self.get_parameters(grad_only)
        nn.save_parameters(path, params)

    def load_parameters(self, path, raise_if_missing=False):
        r"""Loads parameters from a file with the specified format.

        Args:
            path (str): The path to file.
            raise_if_missing (bool, optional): Raise exception if some
                parameters are missing. Defaults to `False`.
        """
        with nn.parameter_scope('', OrderedDict()):
            nn.load_parameters(path)
            params = nn.get_parameters(grad_only=False)
        self.set_parameters(params, raise_if_missing=raise_if_missing)

    def extra_format(self):
        r"""Set the submodule representation format.
        """
        return '.{}'

    def extra_repr(self):
        r"""Set the extra representation for the module."""
        return ''

    def __str__(self):
        r"""Return str representtation of the module."""
        main_str = f'{self.__class__.__name__}(' + self.extra_repr()
        sub_str = ''
        for key, module in self.modules.items():
            m_repr = str(module).split('\n')
            head = [self.extra_format().format(key) + ': ' + m_repr.pop(0)]
            tail = [m_repr.pop()] if len(m_repr) else []
            m_repr = [' '*2 + line for line in (head + m_repr + tail)]
            sub_str += '\n' + '\n'.join(m_repr)
        main_str += sub_str + ('\n' if sub_str else '') + ')'
        return main_str

    def __call__(self, *args, **kargs):
        self.input_shapes = [x.shape for x in args]
        return self.call(*args, **kargs)

    def call(self, *args, **kargs):
        r"""Implement the call of module. Inputs should only be Variables."""
        raise NotImplementedError
