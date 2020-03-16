Logging and visualization
-------------------------

NNablaNAS provides a numerous tools to create visualizations and logging files based on the search space results. Most of these features are integrated into  `tensorboard <https://www.tensorflow.org/tensorboard>`_. Before going further, more details on TensorBoard can be found at `<https://www.tensorflow.org/tensorboard/>`_.

Once tensorboard is installed, we can write scalars, images, and graphs into a directory for visualization within the Tensorboard UI. To run tensorboard, use the following command:

::

    tensorboard --logdir=./log


Logging
.......

All training curves during searching and retraining are logged for one experiment. Users can easily get access to these learning curves by running tensroboard.

.. image:: ../images/logging.png
    :width: 800
    :align: center

**Fig. 1.** An example of training and validation curves.

Architecture visualization
..........................

For some search space, we have provided functions to visualize the architecture learned during the searching procedure.

.. image:: ../images/darts_normal.png
    :width: 600
    :align: center

**Fig. 2.** A normal cell learned by DARTS algorithm.


Graph visualization
...................

One of the strength features in NNablaNAS is its ability to visualize the whole computational graph. This is quite useful for debuging complex model architectures. 


.. image:: ../images/tensorboard.png
    :width: 600
    :align: center
