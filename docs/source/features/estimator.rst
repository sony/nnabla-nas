Estimating the Latency of DNN Architectures
===========================================

Hardware aware NAS addresses the problem, how to fit the architecture
of DNNs to specific target devices, such that they fulfill given
performance requirements. This is for example important if we want to deploy
DNN based algorithms to mobile devices. Naturally, we want to find
DNN architectures that run fast and require require only little memory.
More specifically, we might be interested in DNNs that have

- a low latency.
- a small parameter memory footprint.
- a small activation memory footprint.
- a high throughput.
- a low power consumption.

To perform hardware aware NAS, we therefore need tools to estimate
such performance measure on target devices.
The module nnabla_nas.utils.estimator implements
such tools, i.e., it provides methods to
estimate the latency and the memory footprint of DNN architectures.



How to estimate the latency of DNN architectures
................................................

There are different ways how to estimate the latency of a DNN architecture
on device. Two naive ways how to do it are given in the figure below, namely

- network based estimation
- layer based estimation

.. image:: images/measurement.png

Here, z is a random vector which encodes the structure of the network.
A network based latency estimator, instantiates and measures the time it takes to
calculate the output of the computational graph at once. We call the resulting
latency the true latency. A layer based estimator
instantiates the computational graph and estimates the latency of each layer separately.
The latency of the whole network is calculated as the sum of all the individual layer latencies.
We call this the accumulated latency. Because the individual calculation of each layer causes some
some computational overhead, the layer based latency estimate is not the same as the true latency.
However, experiments show that the differences between the true and the accumulated latency estimates
are small, meaning that both can be used for hardware aware NAS.

In the NNabla NAS framework, we only implement layer based latency estimators. The reason
for this is, that we want the estimators to run offline, i.e., before the architecture
search. Depending on the target hardware, a latency measurement on device can take
considerable time. Therefore, latency measurements during architecture search are
not desireable. With network based estimators, the number of networks to measure grows
exponentially with the number of layers and the number of candidates per layer. However,
with the layer based approach, the growth is only linear.



How to use the estimators
.........................

The following example shows how to use the an estimator. First, we instantiate the model
we want to estimate the latency of. To this end, we borrow the implementation of the
MobileNet from nnabla_nas.contrib.mobilenet. If the network is constructed from dynamic modules,
the nnabla graph must be constructed once, such that each module knows its input shapes. We can then
feed the model to the estimator to calculate the latency. Please note, the estimator
always assumes a batch size of one. Further, the model will always be profiled with the input shapes
that have been calculated when the last nnabla graph was created.

.. code-block:: python

    from nnabla_nas.contrib.mobilenet import TrainNet
    from nnabla_nas.utils.estimator import LatencyEstimator
    import nnabla as nn
    from nnabla.ext_utils import get_extension_context

    cuda_device_id = 0
    ctx = get_extension_context('cudnn', device_id=cuda_device_id)
    nn.set_default_context(ctx)

    inp = nn.Variable((1,3,32,32))
    net = TrainNet()
    #create the nnabla graph once (this defines the input shapes of all modules)
    out = net(inp)

    est = LatencyEstimator()
    latency = est.get_estimation(net)


Please note, if the candidate space contains zero modules, the estimate can deviate considerably
if the model is constructed from dynamic modules. To make this clearer, we continue the
code example from above.

.. code-block:: python

    inp = nn.Variable((1,3,128,128))
    out2 = net(inp)
    latency2 = est.get_estimation(net)

Because we constructed a second nnabla graph (out2) that has a much larger input,
the input shapes of all modules in net will be changed accordingly. Therefore,
latency2 will be much larger then the previously measured latency. Profiling static graphs is similar.
The only difference is, that the input shapes of static modules cannot change after instantiation, meaning
that we do not need to construct a nnabla graph before latency estimation.
