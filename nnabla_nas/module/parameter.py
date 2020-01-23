import nnabla as nn
import numpy as np


class Parameter(nn.Variable):
    """Parameter is a Variable.
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
            # If initializer is not set, returns a new variable with zeros.
            obj.data.zero()
        elif isinstance(initializer, np.ndarray):
            # Initialize by a numpy array.
            assert tuple(shape) == initializer.shape
            obj.d = initializer
        elif callable(initializer):
            # Initialize by Initializer or callable object
            obj.d = initializer(shape=list(map(int, shape)))
        else:
            # Invalid initialzier argument.
            raise ValueError(
                '`initializer` must be either the: obj: `numpy.ndarray`'
                'or an instance inherited from'
                '`nnabla.initializer.BaseInitializer`.'
            )
        obj.grad.zero()
        return obj

    def __repr__(self):
        return 'Parameter containing: ' + super().__repr__()
