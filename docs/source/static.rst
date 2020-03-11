Overview Static Modules
=======================

Besides (dynamic) modules NNablaNAS offers static_modules, i.e., 
modules that can be used to define static computational graphs. 
Although the dynamic network graph implementation has proven to 
be a powerfull tool for many deep learning applications, 
it lacks some features that are needed for 
hardware aware neural architecture search.

With dynamic network graphs, the graph structure is completely defined
in the code, but not encoded in the data structure. Therefore, a dynamic graph 
definition is not the natural choice if we need to define functions that 
need knowledge about the graph structure. Consider the following code example
that defines a simple d layer CNN:

.. code-block:: python
   from nnabla_nas import module as Mo
   import nnabla as nn
 
   inp = nn.Variable((10, 3, 32, 32))

   def net(x, d=10):
      c_inp = Mo.Conv(3, 64, (32,32))
      c_l = [Mo.Conv(64, 64, (32,32)) for i in range(d-1)] 
      x = c_inp(x) 

      for i in range(d-1):
         x = c_l[i](x)
      return x

   out = net(inp)

The network consists of 3 convolutional layers, with a 3x3 kernel. Each layer
computes 64 feature maps. Following the dynamic graph paradigm, 
the structure of the network is only defined in the code, i.e., it is only defined
by the sequence in which we apply the layers c_l. The modules themselves are agnostic to
the graph structure, i.e., they do not know which module is their parents or which
input and output shapes they should expect. 

A dynamic graph definition is not the natural choice if we need to define functions that 
need knowledge about the graph structure. In case of hardware aware NAS, such functions are
for example, latency estimation given the graph structure, the calculation of 
graph similarities (Bayesian Optimization) or simple graph optimization algorithms (as discussed later).
NNablaNAS therefore also offeres static_modules. Static modules are a simple extension of 
dynamic modules and inherit all of their functionality. In comparison, static modules 
store the graph structure and therefore can be used to define static network graphs. 
The example network from the example above can for example be defined like:

..code-block:: python
   from nnabla_nas.module import static_module as Smo
   import nnabla as nn

   def net(x, d=10):
      modules = [Smo.Input(nn.Variable((10, 3, 32, 32)))]
      for i in range(d):
         modules.append(Smo.Conv(parents=[modules[-1]], modules[-1].shape[1], 64, (32,32)))
      return modules[-1]

   out = net()

In comparison to dynamic modules, each static module keeps a list of its parents. Therefore, the graph
structure is stored within and can later be retrieved from the modules. 
Furthermore, static_modules introduce a sort of shape security, i.e.,
once a module is instantiated, the input and output shape of the module are fixed and cannot be changed
anymore.

Why Static Modules for hardware aware NAS
=========================================
There are multiple reasons, why static modules are interesting for hardware aware NAS. Here, we discuss two 
particulary important ones.

Typically, hardware aware NAS involves the definition of large candidate spaces, i.e., 
large DNN architectures that are contain all kind of candidate layers that are
heavily interconnected. During architecture search we consecutively draw subnetworks
from the candidate space, meaning that some of the candidate layers are selected,
while others are dropped. For an efficient search, it is desireable to have simple
graph optimization algorithms in place, i.e., algorithms which optimize the computational 
graph of the selected subnetworks before executing them.

Consider for example the following search space: 1) The network applies an input convolution (conv 1). 2) Two candidate
layers are applied to the output of conv 1, that are a zero operation and another convolution (conv 2). 3) The Join layer
randomly selects the output of one of the candidate layers and feeds it to conv 3. If Join selects Conv 2, we need to calculate
the output of Conv 1, Conv 2 and Conv 3. However, if Join selects Zero, only the output of Conv 3 must be calculated, because
selecting Zero, effectively cuts the computational graph, meaning that all layers that are parent of Zero and that have
no shortcut connection to any following layer can be deleted from the computational graph.
Static modules implement such a graph optimization, meaning that they can speed up computations.

..image:: ../sources/images/static:example_graph.png

A second reason why a static graph definition is the natural choice for hardware aware NAS is related to latency modeling. 
To perform hardware aware NAS, we need to estimate the latency of the subnetworks that have been
drawn from the candidate space in order to decide whether the network meets our latency requirements or not.
Typically, the latency of all layers (modules) within the search space are measured once individually. The latency of a 
subnetwork of the search space, then, is a function of those individual latencies and of the structure of the subnework. Note,
simply summing up all the latencies of the modules that are contained in the subnetwork is wrong. This is obvious if we reconsider the
example from above. All the modules Conv 1 to Conv 3 have a latency > 0, while Zero and Join have a latency of 0. If Join selects Zero,
Conv 1, Zero, Join and Conv 3 are part of the subnetwork. However, summing upt the latency of Conv 1, 
Zero, Join and Conv 3 is wrong. The correct latency would be if we only consider Conv 3.

Other problems which need knowledge of the graph structure are for example:
1) Graph similarity calculation 
2) NAS, using Bayesian optimization algorithms
3) Modelling the memory footprint of DNNs (activation memory) 
  
Which modules are currently implemented
=======================================
There is a static version of all dynamic modules implemented in nnabla_nas.modules. There are currently two static search spaces,
namely contrib.zoph and the contrib.random_wired.
