[![Build status](https://github.com/nnabla/nnabla-nas/workflows/Build%20nnabla-nas/badge.svg)](https://github.com/nnabla/nnabla-nas/actions)

<img align="center" src="docs/source/logo/logo.png" alt="drawing" width="600"/>

# Neural Architecture Search for Neural Network Libraries

NNablaNAS is a Python package that provides methods for neural hardware aware neural architecture search for NNabla

- A top-level graph to define candidate architectures for convolutional neural networks (CNNs)
- Profilers to measure the hardware demands of neural architectures (latency, number of parameters, etc...)
- Searcher algorithms to learn the architecture and model parameters (e.g., [`DartsSearcher`](nnabla_nas/runner/searcher/darts.py) and [`ProxylessNasSearcher`](nnabla_nas/runner/searcher/pnas.py))
- Regularizers (e.g., [`LatencyEstimator`](nnabla_nas/contrib/estimator/latency.py) and [`MemoryEstimator`](nnabla_nas/contrib/estimator/memory.py)) which can be used to enforce hardware constraints


NNablaNAS aims to make the architecture search research more reusable and reproducible by providing them with a modular framework that they can use to implement new search algorithms and new search spaces while reusing code.

- [Neural Architecture Search for Neural Network Libraries](#neural-architecture-search-for-neural-network-libraries)
  - [Getting started](#getting-started)
    - [Installation](#installation)
    - [Setup the datasets](#setup-the-datasets)
    - [Examples](#examples)
  - [Features](#features)
    - [Search spaces](#search-spaces)
    - [Searcher algorithms](#searcher-algorithms)
    - [Logging](#logging)
    - [Visualization](#visualization)
  - [Experiments](#experiments)
  - [Documentation](#documentation)
  - [Contact](#contact)
  - [License](#license)

## Getting started

Here we show how to install NNablaNAS and build a simple search space.

### Installation

It is generally a good idea to install into a Python virtual environment
which provides isolation from system packages.
```
python -m venv venv && source venv/bin/activate
```

A release versions of NNablaNAS may then be installed from PyPI.
```
python -m pip install --upgrade pip
python -m pip install nnabla-nas
```

Or clone the latest development version and install as editable package.
```
git clone git@github.com:sony/nnabla-nas.git && cd nnabla-nas
python -m pip install --upgrade pip
python -m pip install --editable .
python -m pip install -r dev-requirements.txt
```

Additionally, NNablaNAS requires the NNabla CUDA extension package. Find
a suitable version on https://pypi.org/search/?q=nnabla-ext-cuda and
install with (replace XXX with the desired version):
```
python -m pip install nnabla-ext-cudaXXX
```

A further requirement is the NVIDIA Data Loading Library (DALI) in a version
that fits the installed CUDA toolkit. See https://github.com/NVIDIA/DALI/releases
for possible installation methods. At the time of writing the pip installation
for CUDA 11 toolkit was:
```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
```

### Setup the datasets

Please follow the instructions below to prepare the datasets.

* ImageNet: https://github.com/sony/nnabla-examples/tree/master/image-classification/imagenet#preparing-imagenet-dataset

### Examples

The example below shows how to use NNablaNAS. 

We construct a search space by relaxing the layer that the network can have. Our search space encodes that the network chooses between Convolution, MaxPooling, and Identity for the first layer.

```python
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
```

The [tutorials](docs/source/tutorials/) and [examples](docs/source/tutorials/examples.rst) cover additional aspects of NNablaNAS.

## Features

The main features of NNablaNAS are 

### Search spaces

Search spaces are constructed using Modules. Modules are composed of layers, which receive NNabla Variable as input and computes Variable as output. Modules can also contain other Modules, allowing to nest them in a tree structure. One can assign the submodules as regular attributes. All search space components should inherit from `nnabla_nas.module.Module` and override the `call()` method. Please refer to [`nnabla_nas/module/module.py`](nnabla_nas/module/module.py).


```python
from nnabla_nas.model import Model

class MyModule(Module):

    def __init__(self):
        # TODO: write your code here

    def call(self, input):
        # TODO: write your code here
```

A search space is defined as a `Model`, which should inherit API from the class [`nnabla_nas.contrib.model.Model`](nnabla_nas/contrib/model.py). The base API for `Model` has two methods, `get_arch_parameters()`
and `get_net_parameters()` that return the architecture parameters and model parameters, respectively.

```python
from nnabla_nas.contrib.model import Model

class MyModel(Model):

    def get_arch_parameters(self, grad_only=False):
        # TODO: write your code here

    def get_net_parameters(self, grad_only=False):
        # TODO: write your code here
```


### Searcher algorithms

A Searcher interacts with the search space through a simple API. A searcher samples a model from the search space by assigning values to the architecture parameters. The results from sampled architecture is then used to update the architecture parameters of the search space. A searcher also updates the model parameters. A new Searcher should inherit API from [`nnabla_nas.runner.searcher.search.Searcher`](nnabla_nas/runner/searcher/search.py). This class has two methods `train_on_batch()` and `valid_on_batch()` which should be redefined by users. 

```python
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
```

There are two searcher algorithms implemented in NNablaNAS, including [`DartsSearcher`](nnabla_nas/runner/searcher/darts.py) and [`ProxylessNasSearcher`](nnabla_nas/runner/searcher/pnas.py).

### Logging

When running the architecture search, the evaluations in the search space are logged. We maintain a folder to keep track of the parameters, predictions (e.g., loss, error, number of parameters, and latency). Users can easily monitor the training curves with [`TensorboardX`](https://tensorboardx.readthedocs.io/en/latest/tutorial.html).

<img align="center" src="docs/source/images/logging.png" alt="drawing" width="700"/>


### Visualization

Visualization is useful for debugging and illustrating the search space. One can easily check whether the search space was built correctly.

<img align="center" src="docs/source/images/darts_normal.png" alt="drawing" width="700"/>


## Experiments

NNablaNAS uses Hydra to create configurations for its runnable experiments.
As such, its general configuration (found in conf/config.yaml) is composed from many smaller configuration files that each configure one specific
aspect of an experiment, like the network, dataloader, hyper-parameters and optimizer. On top of that, the general configuration
is also composed of another configuration file called args, which contains arguments that would typically be used in a command
line interface utility:

```bash
    - args.context, to set the extension module (set to 'cudnn' by default)
    - args.device_id, which can be set to any device id to use (set to '-1' by default)
    - args.type_config, the type configuration (set to float by default)
    - args.search, to specify whether a search algorithm is being performed (set to false by default)
    - args.algorithm, the algorithm which is set to run (set to DartsSearcher by default)
    - args.output_path, the output path of the job relative to the default hydra output path (set to '.' by default)
    - args.save_nnp, whether to store the network and parameter with nnp format (set to false by default)
    - args.no_visualize, to disable the visualization with graphviz (set to true by default)
```

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

```bash
	python main.py experiment=classification/darts/cifar10_search
```

The retraining script can be used as

```bash
	python main.py experiment=classification/darts/cifar10_train
```

NNablaNAS also supports multi GPUs. More information can be found `here <https://nnabla.readthedocs.io/en/latest/python/tutorial/multi_device_training.html>`_. Below is an example of searching an architecture with 4 GPUs.

```bash 
    mpirun -n 4 python main.py experiment=classification/darts/cifar10_search
```

The output directory which Hydra generates automatically and where it puts the job output can accessed through hydra.run.dir in the command line (or somewhere in the config file), and be modified as follows:

```bash
python main.py experiment=my_experiment hydra.run.dir=new_output_directory
```

## Documentation

To build documentation in various formats, you will need [Sphinx](http://www.sphinx-doc.org) and the
readthedocs theme.

```
cd docs/
pip install -r requirements.txt
```
You can then build the documentation by running ``make <format>`` from the
``docs/`` folder. Run ``make`` to get a list of all available output formats.

## Contact
NNablaNAS is currently maintained by [Sony R&D Center, Stuttgart Laboratory 1](https://www.sony.com/en/SonyInfo/research/about/stuttgart-laboratory1/).
For bug reports, questions, and suggestions, use [Github issues](https://github.com/nnabla/nnabla-nas/issues).

## License

NNablaNAS is Apache-style licensed, as found in the [LICENSE](LICENSE) file.
