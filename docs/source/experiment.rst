Experiments
===========

NNablaNAS uses Hydra to create configurations for its runnable experiments.
As such, its general configuration (found in conf/config.yaml) is composed from many smaller configuration files that each configure one specific
aspect of an experiment, like the network, dataloader, hyper-parameters and optimizer. On top of that, the general configuration
is also composed of another configuration file called args, which contains arguments that would typically be used in a command
line interface utility:

::

    - args.context, to set the extension module (set to 'cudnn' by default)
    - args.device_id, which can be set to any device id to use (set to '-1' by default)
    - args.type_config, the type configuration (set to float by default)
    - args.search, to specify whether a search algorithm is being performed (set to false by default)
    - args.algorithm, the algorithm which is set to run (set to DartsSearcher by default)
    - args.output_path, the output path of the job relative to the default hydra output path (set to '.' by default)
    - args.save_nnp, whether to store the network and parameter with nnp format (set to false by default)
    - args.no_visualize, to disable the visualization with graphviz (set to true by default)

The general configuration file also contains information about specific Hydra parameters, namely the output directory in which Hydra should store
the job output in case of a run or a multirun.

Since Hydra effectively turns every single element of the selected configuration into an optional command line argument, everything from the experiment
configuration to Hydra's own configuration can be modified through the command line.

To run an experiment with NNablaNAS, one should create a configuration file (in ``yaml`` format). Example configurations
that run the DARTS, PNAS, FairNAS and OFA algorithms for classification tasks on various search spaces are given in the conf/experiment folder.
The configuration files for the experiments are compositions from the configuration files of: 1) The definition of the dataset. 2) The training
parameters. 3) The search space definition. 4) The parameters of the optimizers that are used to update the architecture and model parameters of the DNN.
For each architecture search, you need to create two separate configuration files, one for the search phase and one for the retraining phase.
A thorough walkthrough one of our experiment configuration files can be found in this documentation under tutorials, examples.

You can start architecture search using `DartsSearcher` by the command below

::

	python main.py experiment=classification/darts/cifar10_search
			   
The retraining script can be used as

:: 

	python main.py experiment=classification/darts/cifar10_train

NNablaNAS also supports multi GPUs. More information can be found `here <https://nnabla.readthedocs.io/en/latest/python/tutorial/multi_device_training.html>`_. Below is an example of searching an architecture with 4 GPUs.

:: 

    mpirun -n 4 python main.py experiment=classification/darts/cifar10_search

The output directory which Hydra generates automatically and where it puts the job output can accessed through hydra.run.dir in the command line (or somewhere in the config file), and be modified as follows:

python main.py experiment=my_experiment hydra.run.dir=new_output_directory
