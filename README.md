
![nnabla Logo](docs/sources/images/nnabla.png)

# Neural Architecture Search for Neural Network Libraries

Neural Architecture Search is a Python package that provides methods for neural hardware aware neural architecture search for NNabla

- A top level graph to define candidate architectures for convolutional neural networks (CNNs)
- Profilers to measure the hardware demands of neural architectures (latency, number of parameters, etc...)
- Searcher algorithms to learn the architecture and model parameters (e.g., `DartsSearcher` and `ProxylessNasSearcher`)
- Regularizers (e.g., `LatencyEstimator` and `MemoryEstimator`) which can be used to enforce hardware constraints


NnablaNAS aims to make the architecture search research more reusable and reproducible by providing them with a modular framework that they can use to implement new search algorithms and new search spaces while reusing code.

- [Neural Architecture Search for Neural Network Libraries](#neural-architecture-search-for-neural-network-libraries)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
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

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

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

The example below shows how to use NnablaNAS to search a good neural architecture.

```python
import nnabla as nn
import nnabla_nas

```

The tutorials and examples cover additional aspects of NnablaNAS.

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

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## License

`nnabla_nas` is Apache-style licensed, as found in the LICENSE file.