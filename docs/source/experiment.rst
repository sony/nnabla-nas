Experiments
===========

NNablaNAS has command line interface utility:

::

    usage: main.py [-h] [--context CONTEXT] [--device-id DEVICE_ID]
                [--type-config TYPE_CONFIG] [--search]
                [--algorithm {DartsSearcher,ProxylessNasSearcher,Trainer}]
                [--config-file CONFIG_FILE] [--output-path OUTPUT_PATH]

    optional arguments:
    -h, --help            show this help message and exit
    --context CONTEXT, -c CONTEXT
                            Extension module. 'cudnn' is highly recommended.
    --device-id DEVICE_ID, -d DEVICE_ID
                            A list of device ids to use, e.g., 0,1,2. This is only valid if you
                            specify `-c cudnn`.
    --type-config TYPE_CONFIG, -t TYPE_CONFIG
                            Type configuration.
    --search, -s          Whether search algorithm is performed.
    --algorithm {DartsSearcher,ProxylessNasSearcher,Trainer}, -a {DartsSearcher,ProxylessNasSearcher,Trainer}
                            Algorithm used to run
    --config-file CONFIG_FILE, -f CONFIG_FILE
                            The configuration file used to run the experiment.
    --output-path OUTPUT_PATH, -o OUTPUT_PATH
                            Path monitoring logs saved.

To run an experiment with NNablaNAS, one should create a configuration file (in ``json`` format). Example configurations 
that run the DARTS and PNAS algorithms for classification tasks on various search spaces are given in the experiments folder.
The configuration files contain: 1) The definition of the dataset. 2) The training parameters. 3) The search space definition. 4) The parameters of the optimizers that are used
to update the architecture and model parameters of the DNN. 
For each architecture search, you need to create two separate configuration files, one for the search phase and one for the retraining phase.
Below is an example configuration file for an architecture search on the CIFAR10 dataset, that uses the DARTS algorithm for NAS.

.. image:: images/experiment_cfg.png
    :width: 500
    :align: center


You can start architecture search using `DartsSearcher` by the command below

::

	python main.py --search \
               -f examples/classification/darts/cifar10_search.json  \
               -a DartsSearcher \
               -o log/classification/darts/cifar10/search
			   
The retraining script can be used as 

:: 

	python main.py -f examples/classification/darts/cifar10_train.json \
               -a Trainer \
               -o log/classification/darts/cifar10/train

NNablaNAS also supports multi GPUs. More information can be found `here <https://nnabla.readthedocs.io/en/latest/python/tutorial/multi_device_training.html>`_. Below is an example of searching an architecture with 4 GPUs.

:: 

    mpirun -n 4 main.py -d 0,1,2,3 --search \
               -f examples/classification/darts/cifar10_search.json  \
               -a DartsSearcher \
               -o log/classification/darts/cifar10/search
