.. _profiler-nnabla:

NNabla
======

Prepare NNabla
--------------

Install NNabla,

::

    pip install -U nnabla nnabla-ext-cuda

or use the docker
`image <https://hub.docker.com/r/nnabla/nnabla-ext-cuda/tags>`__

::

    docker pull nnabla/nnabla-ext-cuda:latest

and use that image for latency measurement.

NAS repository
--------------

.. code:: bash

    git clone https://gitlab.stc.eu.sony.com/bacnguyencong/nnabla_nas
    export PYTHONPATH=${PWD}/nnabla_nas:$PYTHONPATH

Go to the profiler directory, 
    
.. code:: bash

   cd nnabla_nas/utils/profiler

**NOTE** The following example is based on MobileNetV-2 Search Space and
CIFAR-10 dataset. Change the settings if one wants to change a search
space and dataset.

Measure latency and Create Latency Table
----------------------------------------

Meaure latency, then create the latency table,

.. code:: bash

    python create_latency_table.py \
        --search-net-config ../../../../examples/mobilenet_cifar10_search.json \
        --table-name MNV2-CIFAR10-space-latency \
        --n-run 100 \
        --time-scale m \

Table looks like

::
   
   "Conv[in_channels=3,out_channels=32,kernel=[3,3],stride=[2,2],pad=[1,1],dilation=None,base_axis=1,group=1,with_bias=False,fix_parameters=False,channel_last=False][[1,3,32,32]]",cudnn:float,0.021567345
   "BatchNormalization[n_features=32,fix_parameters=False,eps=1e-05,decay_rate=0.9][[1,32,16,16]]",cudnn:float,0.01301527
   "ReLU6[][[1,32,16,16]]",cudnn:float,0.010151863
   "Conv[in_channels=32,out_channels=32,kernel=[3,3],stride=[1,1],pad=[1,1],dilation=None,base_axis=1,group=32,with_bias=False,fix_parameters=False,channel_last=False][[1,32,16,16]]",cudnn:float,0.015711784
   "Conv[in_channels=32,out_channels=16,kernel=[1,1],stride=[1,1],pad=None,dilation=None,base_axis=1,group=1,with_bias=False,fix_parameters=False,channel_last=False][[1,32,16,16]]",cudnn:float,0.02166748
   "BatchNormalization[n_features=16,fix_parameters=False,eps=1e-05,decay_rate=0.9][[1,16,16,16]]",cudnn:float,0.012667179
   "Conv[in_channels=16,out_channels=48,kernel=[1,1],stride=[1,1],pad=[0,0],dilation=None,base_axis=1,group=1,with_bias=False,fix_parameters=False,channel_last=False][[1,16,16,16]]",cudnn:float,0.019187927
   "BatchNormalization[n_features=48,fix_parameters=False,eps=1e-05,decay_rate=0.9][[1,48,16,16]]",cudnn:float,0.012836456


In general, the format is the following.

::

    <ModuleUID>,<Runtime>,<Value>

-  ModuleUID: Module Unique ID to measure latency
-  Runtime: Runtime like cpu:float, cudnn:flot, cuda:half
-  Value: latency (**milli** second)

.. code:: bash

    python create_latency_estimator.py \
        --search-net-config ../../../../examples/mobilenet_cifar10_search.json \
        --latency-table-json MNV2-CIFAR10-space-latency.json \
        --n-run 100 \
        --time-scale m \
        --num-trials 100

One can find the estimator.py with the scale and bias being encoded,
which is used for the estimation of a latency.
