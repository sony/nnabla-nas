How to implement a new search algorithm?
----------------------------------------

A Searcher interacts with the search space through a simple API. A searcher samples a model from the search space by assigning values to the architecture parameters. The results from sampled architecture are then used to update the architecture parameters of the search space. A searcher also updates the model parameters. A new Searcher should inherit API from ``nnabla_nas.runner.searcher.search.Searcher``. This class has two methods ``train_on_batch()`` and ``valid_on_batch()`` which should be redefined by users. For further modification, we also provide two methods ``callback_on_start()`` and ``callback_on_finish()``, which will be called at the beginning and at the end of the training, respectively.

.. code-block:: python

    from nnabla_nas.runner.searcher.search import Searcher

    class MyAlgorithm(Searcher):

        def callback_on_start(self):
            # TODO: write your code here
            
        def train_on_batch(self, key='train'):
            # TODO: write your code here
        
        def valid_on_batch(self):
            # TODO: write your code here
        
        def callback_on_finish(self):
            # TODO: write your code here


There are two searcher algorithms implemented in NNablaNAS, including :ref:`darts-label` and :ref:`pnas-label`.