Neural Architecture Search for NNabla
======================================

NNablaNAS is a Python package that provides methods in neural architecture search for NNabla

- A top-level graph to define candidate architectures for convolutional neural networks (CNNs)

- Profilers to measure the hardware demands of neural architectures (latency, number of parameters, etc...)

- Searcher algorithms to learn the architecture and model parameters (e.g., *DartsSearcher* and *ProxylessNasSearcher*)

- Regularizers (e.g., *LatencyGraphEstimator* and *MemoryEstimator*) which can be used to enforce hardware constraints

In this document, we will describe how to use the Python APIs, some examples, and the contribution guideline for developers. The latest release version can be installed from  `here <https://github.com/sony/nnabla-nas>`_.



Overview
========

.. toctree::
   :maxdepth: 2

   introduction
   installation
   features
   experiment


Tutorials
=========

.. toctree::
   :maxdepth: 2

   tutorial

API documentation
=================

.. toctree::
   :maxdepth: 2
   
   nnablanas_api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
