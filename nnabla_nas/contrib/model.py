from collections import OrderedDict

import nnabla as nn

from .. import module as Mo


class Model(Mo.Module):
    r"""This class is a base `Model`. Your model should be based on this
    class.
    """

    def get_net_parameters(self, grad_only=False):
        r"""Returns an `OrderedDict` containing all network parmeters of the model.

        Args:
            grad_only (bool, optional): If sets to `True`, then only
                parameters with `need_grad=True` will be retrieved. Defaults
                to `False`.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError

    def get_arch_parameters(self, grad_only=False):
        r"""Returns an `OrderedDict` containing all architecture parameters of
            the model.

        Args:
            grad_only (bool, optional): If sets to `True`, then only
                parameters with `need_grad=True` will be retrieved. Defaults
                to `False`.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def summary(self):
        r"""Returns a string summarizing the model."""
        return ''

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
