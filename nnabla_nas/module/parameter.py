import nnabla as nn
import numpy as np


class Parameter(nn.Variable):
    """
    Parameter is a Variable
    """
    def __new__(cls, shape, need_grad=True, initializer=None):
        assert shape is not None
        obj = super().__new__(cls, shape, need_grad)
        obj.grad.zero()

        # If initializer is not set, just returns a new variable with zeros.
        if initializer is None:
            obj.data.zero()  # Initialize with zero.
            return obj

        # Initialize by a numpy array.
        if isinstance(initializer, np.ndarray):  # numpy init
            assert tuple(shape) == initializer.shape
            obj.d = initializer
            return obj

        # Initialize by Initializer or callable object
        if callable(initializer):
            obj.d = initializer(shape=list(map(int, shape)))
            return obj

        # Invalid initialzier argument.
        raise ValueError(
            "`initializer` must be either the :obj:`numpy.ndarray`"
            " or an instance inherited from"
            " `nnabla.initializer.BaseInitializer`.")
