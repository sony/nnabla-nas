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

The MobileNet search space
^^^^^^^^^^^^^^^^^^^^^^^^^^
In this example we use MobileNetV2 :cite:`sandler2018mobilenetv2` as a backbone to build the search space. In :cite:`sandler2018mobilenetv2`, the proposed architecture use fixed inverted bottleneck convolution with a expension factor of 6 and a kernel size of 3x3 (in the rest of this section we will use 3 to define a kernel size of 3x3 for simplicity). Furthermore the number of inverted bottleneck convolution for black (same feature size) is defined. In this example we want to add flexibility in the MobileNetV2 architecture to choose the depth for each block as well as the expension factor and the kernel size for each inverted residual convolution. 

We use the PNAS search algorithm to find a good architecture in this search space. Specifically, the algorithm can choose, for each layer, between different inverted residual convolution settings or to skip the layer (using identity module). Not that a similar experiment was perform in the original PNAS paper :cite:`cai2018proxylessnas`

Using ``NNABLA NAS`` we can find better architectures than the reference MobileNetV2 both for CIFAR10 and for ImageNet.



Running your first example
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
   Before starting you should get the NNABLA NAS framework and install the dependencies. Please refer to the :ref:`installation` documentation. 

First we will see how to search a model  proxyless   

.. bibliography:: references.bib