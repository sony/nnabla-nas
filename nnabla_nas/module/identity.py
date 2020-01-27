from .module import Module


class Identity(Module):
    r"""Identity layer.
    A placeholder identity operator that is argument-insensitive.
    """

    def __init__(self, *args, **kwargs):
        Module.__init__(self)

    def call(self, input):
        return input
