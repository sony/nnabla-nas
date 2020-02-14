
![nnabla Logo](docs/sources/images/nnabla.png)

# Neural Architecture Search for Neural Network Libraries

Neural Architecture Search is a Python package that provides methods for neural hardware aware neural architecture search for NNabla

- A top level graph to define candidate architectures for convolutional neural networks (CNNs)
- Profilers to measure the hardware demands of neural architectures (latency, number of parameters, etc...)
- Searcher algorithms to learn the architecture and model parameters (e.g., [`DartsSearcher`](nnabla_nas/runner/searcher/darts.py) and [`ProxylessNasSearcher`](nnabla_nas/runner/searcher/pnas.py))
- Regularizers (e.g., [`LatencyEstimator`](nnabla_nas/contrib/estimator/latency.py) and [`MemoryEstimator`](nnabla_nas/contrib/estimator/memory.py)) which can be used to enforce hardware constraints


NnablaNAS aims to make the architecture search research more reusable and reproducible by providing them with a modular framework that they can use to implement new search algorithms and new search spaces while reusing code.

- [Neural Architecture Search for Neural Network Libraries](#neural-architecture-search-for-neural-network-libraries)
  - [Getting started](#getting-started)
    - [Dependencies](#dependencies)
    - [Installation](#installation)
    - [Examples](#examples)
  - [Features](#features)
    - [Search Spaces](#search-spaces)
    - [Searcher Algorithms](#searcher-algorithms)
    - [Logging](#logging)
    - [Visualization](#visualization)
  - [Documentation](#documentation)
  - [Contribution Guide](#contribution-guide)
  - [License](#license)

## Getting started



### Dependencies

- numpy
- sklearn
- graphviz
- nnabla
- nnabla-ext

### Installation

For a local installation, run the following code snippet:

```bash
git clone git@gitlab.stc.eu.sony.com:bacnguyencong/nnabla_nas.git
cd nnabla_nas
```

Install NnablaNAS by the following command

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

### Search Spaces

### Searcher Algorithms

### Logging

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

## Contribution Guide

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

NnablaNAS is Apache-style licensed, as found in the [LICENSE](LICENSE) file.