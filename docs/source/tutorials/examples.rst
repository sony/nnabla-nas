NNABLA NAS examples
--------------------

``NNABLA NAS`` contains over 10 examples examples  including:
 * ``DARTS`` search space abd search algorithm :cite:`liu2018darts`
 * ``Proxyless NAS`` (PNAS) algorithm with mobilenet search space :cite:`cai2018proxylessnas`
 * ``Zoph`` search space (can be searched with DARTS or PNAS algorithms) :cite:`zoph2016neural`
 * ``Randomly wired neural network`` :cite:`xie2019exploring`

 The examples can be launched from a unique entry point ``./main.py`` and all the configurations of each experiment are predefined in some ``json`` files. The list of command lines to run the prepared examples can be found in ``./jobs.sh``. 

In this tutorial we will see how to run the examples and how you can modify the configurations to run your own experiments.

We will show how to run and modify the so-called ``MobileNet`` example on CIFAR10. After running the MobileNet example we advise to try the other examples on your own.  
.. note::
   Commend line for each example can be found in ``./jobs.sh``

The MobileNet search space
^^^^^^^^^^^^^^^^^^^^^^^^^^
In this example we use MobileNetV2 :cite:`sandler2018mobilenetv2` as a backbone to build the search space. In :cite:`sandler2018mobilenetv2`, the proposed architecture use fixed inverted bottleneck convolution with a expension factor of 6 and a kernel size of 3x3 (in the rest of this section we will use 3 to define a kernel size of 3x3 for simplicity). Furthermore the number of inverted bottleneck convolution for black (same feature size) is defined. In this example we want to add flexibility in the MobileNetV2 architecture to choose the depth for each block as well as the expension factor and the kernel size for each inverted residual convolution. 

We use the PNAS search algorithm to find a good architecture in this search space. Specifically, the algorithm can choose, for each layer, between different inverted residual convolution settings or to skip the layer (using identity module). Not that a similar experiment was perform in the original PNAS paper :cite:`cai2018proxylessnas`

Using ``NNABLA NAS`` we can find better architectures than the reference MobileNetV2 both for CIFAR10 and for ImageNet.
 
Running your first example
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
   Before starting you should get the NNABLA NAS framework and install the dependencies. Please refer to the :ref:`installation` documentation. 

First we will run the search with the default setting::

      python main.py -d 0 --search 
                     -f examples/mobilenet_cifar10_search.json \
                     -a ProxylessNasSearcher \
                     -o log/mobilenet/cifar10/search

We used the following arguments:
 * ``main.py`` is the entry script for all search and training examples. 
 * ``-d 0`` indicates the GPU device 0 will be used.
 * ``-f examples/mobilenet_cifar10_search.json`` points to a json file describing the experiment configuration (we will look into the configuration later in this tutorial).
 * ``-a ProxylessNasSearcher`` to use the PNAS algorithm.
 * ``-o log/mobilenet/cifar10/search`` gives the output path to save the logs, models, etc. 

 Not that the device number and the algorithm could be set directly in the json file. In this case you can omit it in the command line. 

The commands run the search using the PNAS algorithm, it will take many hours to run. While it is running, let's have a look at the output path. In ``./log/mobilenet/cifar10/search`` you will find the following files:
* ``arch.h5`` it contains the best architecture so far.
* ``arch.png`` to visualize the best architecture so fare. 
* ``config.json`` save the configuration used for this experiment
* ``log.txt`` contains the search log
Here is an example of an MobileNet architecture:

.. image:: images/arch.png
    :width: 600
    :align: center 

You can also monitor the search using the tensorboard. To run tensorboard, use the following command:

::

    tensorboard --logdir=./log

Access your tensorboard page using your browser at the given adress (typically: `<http://localhost:6006/>`)

Once the search is finished, retrain the winning architecture from scratch using the same entry point python script::

   python main.py -d 0 \
                  -f examples/mobilenet_cifar10_train.json \
                  -a Trainer \
                  -o log/mobilenet/cifar10/train

Note that, this time, we use the ``Training`` algorithm. The retraining will take several hours. You can monitor the training from your tensorboard.

If you want to compare with the original implementation of MobileNetV2, just run::

   python main.py -d 1\
                  -f examples/mobilenet_cifar10_reference.json  \
                  -a Trainer \
                  -o log/mobilenet/cifar10/reference
Congratualtions, you have performed your first meural architecture searach using NNABLA NAS. Now let's have a look how to customize the search and training configuration. 

Search Configuration
^^^^^^^^^^^^^^^^^^^^

Without writing any python code, you can flexibly change the search configuration. Let's go through ``examples/mobilenet_cifar10_search.json``::
   
    "dataset": "cifar10",
    "epoch": 150,
    "batch_size_train": 256,
    "batch_size_valid": 256,
    "mini_batch_train": 64,
    "mini_batch_valid": 64,
    "warmup": 50,
    "cutout": 16,
    "print_frequency": 20,
    "train_portion": 0.9,

These are the runner arguments. ``dataset``, ``epoch`` are self-explanatory. ``batch_size_train`` is the batch size used for training and ``mini_batch_train`` specifies the number of examples transfer onto the GPU at one time. The gardients of the ``mini_batch_train`` are cumulated before updating the model. Keep ``mini_batch_train`` to the same value of ``batch_size_train`` if you have enough GPU memory but it is useful to set a lower ``mini_batch_train`` so that the mini batch can fit in GPU memory while still doing the update on a larger batch. ``batch_size_valid`` and ``mini_batch_valid`` set the correcponding batch size and mini batch size for the validation. 

Before starting updating the architecture, it is benificial to warmup the model parameters. The number of warmup epoch is defined with the ``warmup`` argument.

Cutout is a simple regularization technique for convolutional neural networks that involves removing contiguous sections of input images, effectively augmenting the dataset with partially occluded versions of existing samples. The ``cutout`` argument specifies the lenth of the region that will be cutout. 

``print_frequency`` set how often the partial results are printed in the log file. 

During search the training data is split into two parts. One part is used to train the model parameters and the other part is used to update the architecture parameters. ``train_portion`` set the portion of the training sample that is used to train the parameters. 









.. bibliography:: references.bib