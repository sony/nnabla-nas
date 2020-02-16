
[![pipeline status](https://gitlab.stc.eu.sony.com/bacnguyencong/nnabla_nas/badges/master/pipeline.svg)](https://gitlab.stc.eu.sony.com/bacnguyencong/nnabla_nas/commits/master)
[![coverage report](https://gitlab.stc.eu.sony.com/bacnguyencong/nnabla_nas/badges/master/coverage.svg)](https://gitlab.stc.eu.sony.com/bacnguyencong/nnabla_nas/commits/master)


![nnabla Logo](docs/sources/images/nnabla.png)

# Neural Architecture Search for Neural Network Libraries

NnablaNAS is a Python package that provides methods for neural hardware aware neural architecture search for NNabla

- A top level graph to define candidate architectures for convolutional neural networks (CNNs)
- Profilers to measure the hardware demands of neural architectures (latency, number of parameters, etc...)
- Searcher algorithms to learn the architecture and model parameters (e.g., [`DartsSearcher`](nnabla_nas/runner/searcher/darts.py) and [`ProxylessNasSearcher`](nnabla_nas/runner/searcher/pnas.py))
- Regularizers (e.g., [`LatencyEstimator`](nnabla_nas/contrib/estimator/latency.py) and [`MemoryEstimator`](nnabla_nas/contrib/estimator/memory.py)) which can be used to enforce hardware constraints


NnablaNAS aims to make the architecture search research more reusable and reproducible by providing them with a modular framework that they can use to implement new search algorithms and new search spaces while reusing code.

- [Neural Architecture Search for Neural Network Libraries](#neural-architecture-search-for-neural-network-libraries)
  - [Getting started](#getting-started)
    - [Installation](#installation)
    - [Examples](#examples)
  - [Features](#features)
    - [Search spaces](#search-spaces)
    - [Searcher algorithms](#searcher-algorithms)
    - [Logging](#logging)
    - [Visualization](#visualization)
  - [Documentation](#documentation)
  - [Contribution](#contribution)
  - [License](#license)

## Getting started

Here we show how to install NnablaNAS and build a simple search space.

### Installation

For a local installation, run the following code snippet:

```bash
git clone git@gitlab.stc.eu.sony.com:bacnguyencong/nnabla_nas.git
cd nnabla_nas
```

Install dependecies for NnablaNAS by the following command

```bash
pip install -r requirements.txt
```

Run tests to check for correctness:

```bash
pytest .
```

### Examples

The example below shows how to use NnablaNAS. 

We construct a search space by relaxing the layer that the network can have. Our search space encodes that the network chooses between Convolution, MaxPooling, and Identity for the first layer.

```python
from collections import OrderedDict

from nnabla_nas import module as Mo
from nnabla_nas.contrib.darts.modules import MixedOp
from nnabla_nas.contrib.model import Model


class MyModel(Model):
    def __init__(self):
        self._block = MixedOp(
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

The [tutorials](docs/tutorial.md) and [examples](docs/examples.md) cover additional aspects of NnablaNAS.

## Features

The main features of NnablaNAS are 

### Search spaces

Search spaces are constructed using Modules. Modules are composed of layers, which receives nnabla Variable as input and computes Variable as output. Modules can also contain other Modules, allowing to nest them in a tree structure. One can assign the submodules as regular attributes. All search space components should inherit from `nnabla_nas.module.Module` and override the `call()` method. Please refer to [`nnabla_nas/module/module.py`](nnabla_nas/module/module.py).

A new model should inherit API from the class [`nnabla_nas.contrib.model.Model`](nnabla_nas/contrib/model.py). The base API for `Model` has two methods, `get_arch_parameters()`
and `get_net_parameters()` that return the architecture parameters and model parameters, respectively.

### Searcher algorithms

An Searcher interacts with the search space through a simple API. A searcher samples a model from the search space by assigning values to the architecture parameters. The results from sampled architecture is then used to update the architecture parameters of the search space. A searcher also updates the model parameters. A new Searcher should inherit API from [`nnabla_nas.runner.searcher.search.Searcher`](nnabla_nas/runner/searcher/search.py). This class has two methods `train_on_batch()` and `valid_on_batch()` which should be redefined by users. 

There are two searcher algorithms implemented in NnablaNAS, including [`DartsSearcher`](nnabla_nas/runner/searcher/darts.py) and [`ProxylessNasSearcher`](nnabla_nas/runner/searcher/pnas.py).

### Logging

When running the architecture search, the evaluations in the sarch space are logged. We mantain a folder to keep track of the parameters, predictions (e.g., loss, error, number of parameters, and latency). Users can easily monitor the training curves with [`TensorboardX`](https://tensorboardx.readthedocs.io/en/latest/tutorial.html).


### Visualization

## Documentation

To build documentation in various formats, you will need [Sphinx](http://www.sphinx-doc.org) and the
readthedocs theme.

```
cd docs/
pip install -r requirements.txt
```
You can then build the documentation by running ``make <format>`` from the
``docs/`` folder. Run ``make`` to get a list of all available output formats.

## Contribution

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

NnablaNAS is Apache-style licensed, as found in the [LICENSE](LICENSE) file.