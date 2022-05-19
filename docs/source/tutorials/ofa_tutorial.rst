Once-for-All tutorial
---------------------

Once-for-All (OFA) :cite:`cai2019once` is only trained once, and we can quickly get specialized sub-networks from the OFA network without additional training.
The OFA search space contains a large number of sub-networks (>10^19) that covers various hardware platforms. 
To efficiently train the search space, the progressive shrinking algorithm enforces the training order from large sub-networks to small sub-networks in a progressive manner.

We will show how to run ``OFA`` example on ImageNet. 

.. note::
    The command line for each example can be found in ./jobs.sh

Progressive shrinking algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we use MobileNetV3-Large :cite:`howard2019searching` based backbone to build the search space.
This search space is called a ``SuperNet`` that includes various sub-networks as the candidate architectures.
The search space includes resolution, kenerl size, depth, and width expansion ratio.
OFA's progressive shrinking algorithm gradually increases the target search space.

.. image:: images/progressive_shrinking.png
    :width: 600
    :align: center 


In NNablaNAS, progressive shrinking can be performed by running the training with different search space in a progressive manner.

Here are the running scripts found in ``./jobs.sh``. ::

    mpirun -n 8 python main.py \
                -f examples/classification/ofa/ofa_imagenet_search_fullnet.json \
                -a OFASearcher \
                -o log/classification/ofa/imagenet/search/K7_E6_D4/

    mpirun -n 8 python main.py --search \
                -f examples/classification/ofa/ofa_imagenet_search_kernel.json \
                -a OFASearcher \
                -o log/classification/ofa/imagenet/search/K357_E6_D4/

    mpirun -n 8 python main.py --search \
                -f examples/classification/ofa/ofa_imagenet_search_depth_phase1.json \
                -a OFASearcher \
                -o log/classification/ofa/imagenet/search/K357_E6_D34/

    mpirun -n 8 python main.py --search \
                -f examples/classification/ofa/ofa_imagenet_search_depth_phase2.json \
                -a OFASearcher \
                -o log/classification/ofa/imagenet/search/K357_E6_D234/

    mpirun -n 8 python main.py --search \
                -f examples/classification/ofa/ofa_imagenet_search_expand_phase1.json \
                -a OFASearcher \
                -o log/classification/ofa/imagenet/search/K357_E46_D234/

    mpirun -n 8 python main.py --search \
                -f examples/classification/ofa/ofa_imagenet_search_expand_phase2.json \
                -a OFASearcher \
                -o log/classification/ofa/imagenet/search/K357_E346_D234/

Each stage uses weights trained in the previous stage for initialization. 

For example, the network configuration for ``elastic kernel`` stage looks like this::
   
   "network": {
        "ofa": {
            "num_classes": 1000,
            "bn_param": [0.9, 1e-5],
            "drop_rate": 0.1,
            "op_candidates": ["MB6 7x7", "MB6 5x5", "MB6 3x3"],
            "depth_candidates": [4],
            "weights": "log/classification/ofa/imagenet/search/K7_E6_D4/weights.h5"
        }
    },


Train Configuration
^^^^^^^^^^^^^^^^^^^^
Once the SuperNet is trained, you can fine-tune sub-networks to further improve their performance.
Let's have a look at the example ``examples/classification/ofa/ofa_imagenet_train_subnet.json``. 
Most of the configuration parameters are the same as for the search json file. 
The only new configuration parameter is::

    "genotype": [5, 2, 9, 9, 6, 4, 2, 1, 7, 7, 8, 9, 8, 3, 9, 9, 8, 4, 3, 1]

``genotype`` is used to provide the architecture configuration for the sub-network you wish to fine-tune.

.. bibliography:: ../bibtex/reference.bib
