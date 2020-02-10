from .. import module as Mo


class Model(Mo.Module):
    r"""This class is a base `Model`. Your model should be based on this
    class.
    """

    def get_net_parameters(self, grad_only=False):
        r"""Returns an `OrderedDict` containing all network parmaeters of the model.

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
        r"""Returns string printed at each epoch of training."""
        return ''
