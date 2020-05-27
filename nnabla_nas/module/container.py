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

import operator
from collections import OrderedDict

from .module import Module
from .parameter import Parameter


def _get_abs_string_index(obj, idx):
    """Get the absolute index for the list of modules"""
    idx = int(operator.index(idx))
    if not (-len(obj) <= idx < len(obj)):
        raise IndexError('index {} is out of range'.format(idx))
    if idx < 0:
        idx += len(obj)
    return str(idx)


class ModuleList(Module):
    r"""Hold submodules in a list. This implementation mainly follows
    the Pytorch implementation.

    Args:
        modules (iterable, optional): An iterable of modules to add.
    """

    def __init__(self, modules=None):
        if modules is not None:
            self += modules

    def append(self, module):
        r"""Appends a given module to the end of the list.

        Args:
            module (Module): A module to append.
        """
        if not isinstance(module, Module):
            ValueError(f'{module} is not an instance of Module.')
        setattr(self, str(len(self)), module)
        return self

    def extend(self, modules):
        r"""Appends modules from a Python iterable to the end of the list.

        Args:
            modules (iterable): An iterable of modules to append.
        """
        for module in modules:
            self.append(module)
        return self

    def insert(self, index, module):
        r"""Insert a given module before a given index in the list.

        Args:
            index (int): An index to insert.
            module (Module): A module to insert.
        """
        if not isinstance(module, Module):
            ValueError(f'{module} is not an instance of Module.')
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
        if not isinstance(module, Module):
            ValueError(f'{module} is not an instance of Module.')
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

    def extra_format(self):
        return '[{}]'

    def extra_repr(self):
        return f'len={len(self)}'


class ParameterList(Module):
    r"""Hold parameters in a list.

    Args:
        parameters (iterable, optional): An iterable of parameters to add.
    """

    def __init__(self, parameters=None):
        if parameters is not None:
            self += parameters

    def append(self, parameter):
        r"""Appends a given module to the end of the list.

        Args:
            parameter (Parameter): A parameter to append.
        """
        if not isinstance(parameter, Parameter):
            ValueError(f'{parameter} is not an instance of Parameter.')
        setattr(self, str(len(self)), parameter)
        return self

    def extend(self, parameters):
        """Extends an iterable of parameters to the end of the list.

        Args:
            parameters (iterable): An iterable of Parameters.
        """
        for parameter in parameters:
            self.append(parameter)
        return self

    def insert(self, index, parameter):
        r"""Insert a given parameter before a given index in the list.

        Args:
            index (int): An index to insert.
            parameter (Parameter): A parameter to insert.
        """
        if not isinstance(parameter, Parameter):
            ValueError(f'{parameter} is not an instance of Parameter.')
        for i in range(len(self), index, -1):
            self.parameters[str(i)] = self.parameters[str(i - 1)]
        self.parameters[str(index)] = parameter
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(list(self.parameters.values())[index])
        index = _get_abs_string_index(self, index)
        return self.parameters[index]

    def __setitem__(self, index, parameter):
        if not isinstance(parameter, Parameter):
            ValueError(f'{parameter} is not an instance of Parameter.')
        index = _get_abs_string_index(self, index)
        self.parameters[str(index)] = parameter

    def __delitem__(self, index):
        if isinstance(index, slice):
            for k in range(len(self.parameters))[index]:
                delattr(self, str(k))
        else:
            delattr(self, _get_abs_string_index(self, index))
        indices = [str(i) for i in range(len(self.parameters))]
        self._parameters = OrderedDict(
            list(zip(indices, self.parameters.values())))

    def __len__(self):
        return len(self.parameters)

    def __iter__(self):
        return iter(self.parameters.values())

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def extra_format(self):
        return '[{}]'

    def extra_repr(self):
        return f'len={len(self)}'


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
