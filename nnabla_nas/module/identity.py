from .module import Module


class Identity(Module):
    r"""Identity layer.
    A placeholder identity operator that is argument-insensitive.
    """
    def call(self, input):
        return input
