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
import numpy as np


class Parameter(nn.Variable):
    r"""Parameter is a Variable.
    A kind of Variable that is to be considered a module parameter. Parameters
    are :class:`~nnabla.Variable` subclasses, that have a very special
    property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters.

    Args:
        shape (tuple of int): The shape of Parameter.
        need_grad (bool, optional): If the parameter requires gradient.
            Defaults to True.
        initializer (:obj:`nnabla.initializer.BaseInitializer` or
            :obj:`numpy.ndarray`): An initialization function to be applied to
            the parameter. :obj:`numpy.ndarray` can also be given to
            initialize parameters from numpy array data. Defaults to None.
    """

    def __new__(cls, shape, need_grad=True, initializer=None):
        assert shape is not None
        obj = super().__new__(cls, shape, need_grad)
        if initializer is None:
            obj.data.zero()
        elif isinstance(initializer, np.ndarray):
            assert tuple(shape) == initializer.shape
            obj.d = initializer
        elif callable(initializer):
            obj.d = initializer(shape=list(map(int, shape)))
        else:
            raise ValueError(
                '`initializer` must be either the: obj: `numpy.ndarray`'
                'or an instance inherited from'
                '`nnabla.initializer.BaseInitializer`.'
            )
        obj.grad.zero()
        return obj

    def __repr__(self):
        return (f'<Parameter({self.shape}, need_grad={self.need_grad})'
                f' at {hex(id(self))}>')
