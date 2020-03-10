How to construct a search space?
--------------------------------

In this tutorial, we will show how to build search spaces in NNablaNas.


Search spaces are constructed using Modules. Modules are composed of layers, which receives nnabla Variable as input and computes Variable as output. Modules can also contain other Modules, allowing to nest them in a tree structure. One can assign the submodules as regular attributes. All search space components should inherit from `nnabla_nas.module.Module` and override the `call()` method.

.. code:: python

    from nnabla_nas.model import Model

    class MyModule(Module):

        def __init__(self):
            # TODO: write your code here

        def call(self, input):
            # TODO: write your code here