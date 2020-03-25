How to implement a new module?
---------------------------------

NNablaNAS defines a set of **Modules**, which are roughly equivalent to neural network layers. A ``Module`` receives input Variables and computes output Variables. Modules also hold internal states such as Variables containing learnable parameters. A set of common layers are defined in NNablaNAS (please to :ref:`modules`). Modules can also contain other Modules, allowing to nest them in a tree structure. One can assign the submodules as regular attributes. All the training weights of Module classes are implemented as ``Parameter`` objects. Whenever a Module is assigned as a member of another Module, the Parameters of the assignee object have also added the Parameters of the object which is being assigned to. This is often referred to as registering Parameters of a Module. If one tries to assign a Variable to a Module object, it will not show up in the ``get_parameters()`` unless it has been defined as ``Parameter`` object.

A new Module should inherit from ``nnabla_nas.module.Module`` and override the ``call()`` method. Below is an example of Module.


.. code:: python

    import nnabla as nn
    from nnabla_nas import module as Mo


    class Net(Mo.Module):

        def __init__(self):
            self.fc = Mo.Linear(10, 5)
            self.coef = Mo.Parameter((5, 1))

        def call(self, input):
            return self.fc(input) * self.coef


    x = nn.Variable((1, 10))
    net = Net()

    print(net)
    print(net(x))


To register a list of Modules, we should use ``ModuleList``. Similarly, a list of Parameters can be registered by wrapping the list inside a ``ParameterList`` class.

.. code:: python

    self.parameters = Mo.ParameterList([
        Mo.Parameter((1, 2)),
        Mo.Parameter((1, 2))
    ])

    self.modules = Mo.ModuleList([
        Mo.Conv(3, 3, (3, 3)),
        Mo.Conv(3, 5, (3, 3))
    ])

A list of parameters inside Module can be retrieved by calling ``get_parameters()``. In order to get all nested submodules, we can use ``get_modules()``.