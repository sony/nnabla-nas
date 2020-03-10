How to construct a search space?
--------------------------------

In this tutorial, we will show how to build search spaces in NNablaNas.


A search space is defined as a ``Model``, which should inherit API from the class ``nnabla_nas.contrib.model.Model``. The base API for ``Model`` has two methods, ``get_arch_parameters()`` and ``get_net_parameters()`` that return the architecture parameters and model parameters, respectively.


.. code:: python

    from collections import OrderedDict

    from nnabla_nas import module as Mo
    from nnabla_nas.contrib.model import Model


    class MyModel(Model):
        def __init__(self):
            self._block = Mo.MixedOp(
                operators=[
                    Mo.Conv(in_channels=3, out_channels=3, kernel=(3, 3), pad=(1, 1)),
                    Mo.MaxPool(kernel=(3, 3), stride=(1, 1), pad=(1, 1)),
                    Mo.Identity()
                ],
                mode='full'
            )
            self._classifier = Mo.Sequential(
                Mo.ReLU(),
                Mo.GlobalAvgPool(),
                Mo.Linear(3, 10)
            )

        def call(self, input):
            out = self._block(input)
            out = self._classifier(out)
            return out

        def get_arch_parameters(self, grad_only=False):
            r"""Returns an `OrderedDict` containing architecture parameters."""
            p = self.get_parameters(grad_only)
            return OrderedDict([(k, v) for k, v in p.items() if 'alpha' in k])

        def get_net_parameters(self, grad_only=False):
            r"""Returns an `OrderedDict` containing model parameters."""
            p = self.get_parameters(grad_only)
            return OrderedDict([(k, v) for k, v in p.items() if 'alpha' not in k])


    if __name__ == '__main__':
        net = MyModel()
        print(net)
