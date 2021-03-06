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

For a local installation, run the following code snippet:

```bash
git clone git@github.com:sony/nnabla-nas.git
cd nnabla_nas
```

Install dependencies for NNablaNAS by the following command

```bash
pip install -r requirements.txt
```

Run tests to check for correctness:

```bash
pytest .
```

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

NNablaNAS has command line interface utility:

```bash
usage: main.py [-h] [--context CONTEXT] [--device-id DEVICE_ID]
               [--type-config TYPE_CONFIG] [--search]
               [--algorithm {DartsSearcher,ProxylessNasSearcher,Trainer}]
               [--config-file CONFIG_FILE] [--output-path OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --context CONTEXT, -c CONTEXT
                        Extension module. 'cudnn' is highly recommended.
  --device-id DEVICE_ID, -d DEVICE_ID
                        A list of device ids to use, e.g., `0,1,2,3`. This is
                        only valid if you specify `-c cudnn`.
  --type-config TYPE_CONFIG, -t TYPE_CONFIG
                        Type configuration.
  --search, -s          Whether it is searching for the architecture.
  --algorithm {DartsSearcher,ProxylessNasSearcher,Trainer}, -a {DartsSearcher,ProxylessNasSearcher,Trainer}
                        Which algorithm to use.
  --config-file CONFIG_FILE, -f CONFIG_FILE
                        The configuration file for the experiment.
  --output-path OUTPUT_PATH, -o OUTPUT_PATH
                        Path to save the monitoring log files.
```

You can start the architecture search using `DartsSearcher` by the command below

```bash
# search DARTS
python main.py --search \
               -f examples/classification/darts/cifar10_search.json  \
               -a DartsSearcher \
               -o log/classification/darts/cifar10/search
```

For re-training, the model using the architecture found in the architecture search, just run
```bash
# train DARTS
python main.py -f examples/classification/darts/cifar10_train.json \
               -a Trainer \
               -o log/classification/darts/cifar10/train
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
NNablaNAS is currently maintained by [SSG-DL group](mailto:STC_EUTEC-SSG-DL@eu.sony.com), [R&D Center Europe Stuttgart Laboratory 1](https://www.sony.net/SonyInfo/technology/about/stuttgart1/). For bug reports, questions, and suggestions, use [Github issues](https://github.com/nnabla/nnabla-nas/issues).

## License

NNablaNAS is Apache-style licensed, as found in the [LICENSE](LICENSE) file.
