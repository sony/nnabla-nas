How to construct a search space?
--------------------------------

In this tutorial, we will show how to define a search space. Basically, a search space is
a huge DNN model that contains many different candidate layers, which can be selected 
or dropped from the network by adjusting the architecture parameters. In NNablaNAS, 
a search space therefore is defined as a ``Model`` class, 
inherits the API from the class ``nnabla_nas.contrib.model.Model``. 

The base API of the ``Model`` class has four methods, 
``get_arch_parameters()``, ``get_net_parameters()``, ``loss()`` and 
``metrics()``, that return the architecture parameters and model parameters, a loss value and
evaluation metrics for a model, respectively. The following example shows how
you can define a simple search space that contains three different layers, namely 
a convolutional, a maximum pooling and an identity layer, that can be selected by the
architecture search algorithm. The candidate layers are followed by a global average pooling 
and a linear classification layer.


.. code:: python

    from collections import OrderedDict

    from nnabla_nas import module as Mo
    from nnabla_nas.contrib.model import Model
    import nnabla.functions as F

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

        def loss(self, outputs, targets, loss_weights=None, *args):
            assert len(outputs) == 1 and len(targets) == 1
            return F.mean(F.softmax_cross_entropy(outputs[0], targets[0]))
			
        def metrics(self, outputs, targets):
            assert len(targets) == 1
            return {"error": F.mean(F.top_n_error(outputs[0], targets[0]))}

    if __name__ == '__main__':
        net = MyModel()
        print(net)
