============
Introduction
============

The success of deep learning is due to its automation of the feature engineering process. This sucess has been shown in many tasks, including s image recognition, speech recognition, and machine translation. By increasing more complex neural architectures, we can further increase the performance of deep learning models [Elsken2018]_. However, most of the neural architectures are desinged manually, making it unscalable in new domains. A promissing direction in automating machine learning is automating architecture engineering, the so-called *neural network architecture search*. Neural network architecture search is closely related to `hyperparameter optimization <https://en.wikipedia.org/wiki/Hyperparameter_optimization>`_ and is a subfield of `automated machine learning <https://en.wikipedia.org/wiki/Automated_machine_learning>`_ (AutoML). **NNablaNAS** is a framework for architecture search in computer vision domain. The main aim is to provide a modular, easy, and extendible toolbox for deep learning practioners. In this section, an overview of neural architecture search is introduced.


.. toctree::
    :maxdepth: 2


What is neural architecture search?
-----------------------------------

**Neural Architecture Search** (NAS) is a technique in machine learning used to automatically learn neural network architectures for a given machine learning task. Let :math:`\theta` and :math:`\alpha` denote the model and network architecture parameters, NAS can be formulated as a bilevel optimization problem:

.. math::

    \underset{\alpha}{\arg\min} &\quad \mathcal{L}_{\text{val}} (\theta^{*}; \alpha) \\
    \text{s.t.} & \quad \theta^{*} = \underset{\theta}{\arg\min} \; \mathcal{L}_{\text{train}} (\theta; \alpha)

where :math:`\mathcal{L}_{\text{train}}` and :math:`\mathcal{L}_{\text{val}}` denote the training and validation loss function, respectively.


The design of modern neural network architectures is driven by multiple different objectives [liu2018]_:

* The neural network should have a reasonably high capacity, i.e., the the family of transfer functions contains arbitrary complex functions which can capture lots of information from training data. 

* Inference should be computationally efficient, i.e., inference only needs a small number of multiplication-accumulation (MAC) operations, or low inference latency.

The design of a good nueral architecture corresponds to find a good balance  between those (often competing) requirements, by selecting and arranging layers in a meaningful way. Compared to the early days of Deep Learning, today, DNNs consist of a broad variety of different different network layers like: *Linear*, *Convolutional*, *Dilated Convolutional*, *Group Convolutional*, *Separable Convolutional* (depth wise, channel wise, spatial), *Pooling*, *Skip Connect*, *Batch Normalization*, etc. Therefore, neural architecture design is a very large combinatorial problem which  is especially hard to solve, because we have only a poor (or almost no) understanding how a specific choice or arrangement of layers effects our requirements. The aim of neural architecture search is to automate architecture design and to directly learn the optimal architecture from the data. This has multiple benefits. We need no expert with lots of experience. We do not need to understand which effect a combination of certain layers yields to our requirements. NAS has the potential to come up with architectures which generalize much better to unseen data than humans, because it can try out many more architectures in the same time. We can optimize the architectures to be resource efficient.


.. image:: images/nas_overview.png
    :width: 800
    :align: center


Fig 1. An overview of neural architecture search. (Image source: [Elsken2018]_)

The main components of NAS include:

- **Search space**: This defines which architectures or types of artificial neural networks can be used.

- **Search algorithm**: This defines approaches used to explore the search space.

- **Performance estimation strategy**: This evaluates the performance of an architecture.


Agorithms in NAS
----------------

ProxylessNAS [Cai2018]_

.. math::

    \max_{\alpha} &\quad \mathbb{E}_{z \sim p_{\alpha}(z)} \big[\text{score}(z, \Phi^{*})\big] \\
    \text{s.t.} & \quad \Phi^{*} = \underset{\Phi}{\arg \min} \quad \text{loss}(z, \Phi)

DARTS [liu2018]_


Code structure
--------------

.. image:: images/high_level_API.png
    :width: 800
    :align: center


.. rubric:: References

.. [liu2018] Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018).

.. [Elsken2018] Elsken, Thomas, Jan Hendrik Metzen, and Frank Hutter. "Neural architecture search: A survey." arXiv preprint arXiv:1808.05377 (2018).

.. [Cai2018] Cai, Han, Ligeng Zhu, and Song Han. "Proxylessnas: Direct neural architecture search on target task and hardware." arXiv preprint arXiv:1812.00332 (2018).

