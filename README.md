
![nnabla Logo](docs/sources/images/nnabla.png)

# Neural Architecture Search (NAS) for NNabla
This toolbox provides methods for neural hardware aware neural architecture search 
for NNabla, i.e., it implements tools for

1. A top level graph to define candidate architectures for convolutional neural networks (CNNs).
2. Profilers to measure the hardware demands of neural architectures (latency, #parameters, ...).
3. Searcher algorithms to learn the architecture and model parameters (DARTS, ProxylessNAS).
4. Regularizers which can be used to enforce hardware constraints.

This package has been tested, using the environment
described [here](/environment.txt).

## Table of contents
* [Getting Started](#Getting-Started)
* [Installation](#Installation)
* [Documentation](#Documentation)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installation

Install `nnabla_nas` by the following command
```
pip install -r requirements.txt
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

## Contribution Guide

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## License

`nnabla_nas` is Apache-style licensed, as found in the LICENSE file.