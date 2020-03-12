.. _profiler-snpe:

SNPE
====

Prepare docker images
---------------------

.. code:: bash

    docker pull docker.ubiq.sony.co.jp:5000/snpe/ubuntu-18.04_snpe-1.33.1.608

or simply work in ``snpe@deatcs001ws551.eu.sony.com`` with the docker
image.

Currently, the profiler for SNPE is based on SNPE SDK 1.33.

NAS repository
--------------

.. code:: bash

    git clone https://gitlab.stc.eu.sony.com/bacnguyencong/nnabla_nas
    export PYTHONPATH=${PWD}/nnabla_nas:$PYTHONPATH

Go to the profiler directory, 
    
.. code:: bash

   cd nnabla_nas/utils/profiler
          

Prepare phone
-------------

Run ``docker.ubiq.sony.co.jp:5000/snpe/ubuntu-18.04_snpe-1.33.1.608``
container,

.. code:: bash

    docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb -v ${HOME}:${HOME} \
        --user 0 \
        -e http_proxy=${http_proxy} -e https_proxy=${https_proxy} \
        --rm \
        docker.ubiq.sony.co.jp:5000/snpe/ubuntu-18.04_snpe-1.33.1.608

Inside the container

.. code:: bash

    conda activate snpe-env
    adb devices

Confirm the follows inside the docker and/or on the phone screen - Allow
access on the phone screen of notification - Allow USB debugging on the
phone screen of notification

**NOTE** The following example is based on MobileNetV-2 Search Space and
CIFAR-10 dataset. Change the settings if one wants to change a search
space and dataset.

Measure latency
---------------

Generate NNPs
~~~~~~~~~~~~~

Outside the docker container,

.. code:: bash

    python generate_nnps.py \
        --search-net-config ../../../../examples/mobilenet_cifar10_search.json \
        --nnp-dir MNV2-CIFAR10-space-nnps

Run SNPE Bench
~~~~~~~~~~~~~~

Inside the container, run

.. code:: bash

    conda activate snpe-env

    python -u nas_snpe_bench.py \
        --nnp-dir MNV2-CIFAR10-space-nnps \
        --devices QV714AE41T \
        --name MNV2-CIFAR10-space-latency \
        --perf-profile sustained_high_performance \
        --model-random-input 50

Make Latency Table
~~~~~~~~~~~~~~~~~~

Outside the docker container,

.. code:: bash

    python create_latency_table.py --space-latency MNV2-CIFAR10-space-latency

You can find MNV2-CIFAR10-space-latency.meta,
MNV2-CIFAR10-space-latency.json and MNV2-CIFAR10-space-latency.csv. For
example, the contens of the csv file look like

::

    "Conv[in_channels=192,out_channels=192,kernel=[5,5],stride=[1,1],pad=[2,2],dilation=None,base_axis=1,group=192,with_bias=False,fix_parameters=False,channel_last=False][[1,192,16,16]]",GPU,Forward Propagate,Min_Time,595
    "Conv[in_channels=192,out_channels=192,kernel=[5,5],stride=[1,1],pad=[2,2],dilation=None,base_axis=1,group=192,with_bias=False,fix_parameters=False,channel_last=False][[1,192,16,16]]",GPU,Forward Propagate,Avg_Time,672
    "Conv[in_channels=192,out_channels=192,kernel=[5,5],stride=[1,1],pad=[2,2],dilation=None,base_axis=1,group=192,with_bias=False,fix_parameters=False,channel_last=False][[1,192,16,16]]",GPU,Forward Propagate,Max_Time,783
    "Conv[in_channels=192,out_channels=192,kernel=[5,5],stride=[1,1],pad=[2,2],dilation=None,base_axis=1,group=192,with_bias=False,fix_parameters=False,channel_last=False][[1,192,16,16]]",GPU,Total Inference Time,Min_Time,623
    "Conv[in_channels=192,out_channels=192,kernel=[5,5],stride=[1,1],pad=[2,2],dilation=None,base_axis=1,group=192,with_bias=False,fix_parameters=False,channel_last=False][[1,192,16,16]]",GPU,Total Inference Time,Avg_Time,700
    "Conv[in_channels=192,out_channels=192,kernel=[5,5],stride=[1,1],pad=[2,2],dilation=None,base_axis=1,group=192,with_bias=False,fix_parameters=False,channel_last=False][[1,192,16,16]]",GPU,Total Inference Time,Max_Time,812
    "Conv[in_channels=192,out_channels=192,kernel=[5,5],stride=[1,1],pad=[2,2],dilation=None,base_axis=1,group=192,with_bias=False,fix_parameters=False,channel_last=False][[1,192,16,16]]",GPU,Layers Time,Min_Time,37.0
    "Conv[in_channels=192,out_channels=192,kernel=[5,5],stride=[1,1],pad=[2,2],dilation=None,base_axis=1,group=192,with_bias=False,fix_parameters=False,channel_last=False][[1,192,16,16]]",GPU,Layers Time,Avg_Time,37.0
    "Conv[in_channels=192,out_channels=192,kernel=[5,5],stride=[1,1],pad=[2,2],dilation=None,base_axis=1,group=192,with_bias=False,fix_parameters=False,channel_last=False][[1,192,16,16]]",GPU,Layers Time,Max_Time,38.0

In general, the format is the following.

::

    <ModuleUID>,<Runtime>,<LatencyKey>,<Stats>,<Value>

-  ModuleUID: Module Unique ID to measure latency
-  Runtime: Runtime like CPU, GPU, DSP and its combination of input
   buffer type
-  LatencyKey: Forward Propagate, Total Inference Time, or Layers Time
-  Stats: Min\_Time, Avg\_Time, Max\_Time
-  Value: latency (**micro** second)

NOTE
^^^^

-  For *Runtime*\ key, use *CPU*, *GPU*, *GPU\_FP16*, or *DSP*.
-  Use a value looked up by
   ``<ModuleUID>,<Runtime>,<Layers Time>,<Avg\_Time>``.
-  All keys come from the results of snpe-bench.py excpet for
   ``Layers Time``.
-  The definition of ``Layers Time`` is the accumulation of the latency
   over the layers in a DLC except for the input layer (SNPE defines the
   input layer for the input).

Create Latency Estimator
~~~~~~~~~~~~~~~~~~~~~~~~

NOTE: snpe\_bench parts are separated since the nnabla\_nas heavily
depends on python>=3.6.

Outside the docker container,

.. code:: bash

    python sample_nnps.py \
        --search-net-config ../../../../examples/mobilenet_cifar10_search.json \
        --latency-table-json MNV2-CIFAR10-space-latency.json \
        --nnp-dir MNV2-CIFAR10-sampled-nnps \
        --num-trials 50

Inside the docker container,

.. code:: bash

    python -u nas_snpe_bench.py \
        --nnp-dir MNV2-CIFAR10-sampled-nnps \
        --devices QV714AE41T \
        --name MNV2-CIFAR10-sampled-latency \
        --perf-profile sustained_high_performance \
        --profiling-level detailed \
        --model-random-input 50

Outside the docker container,

.. code:: bash

    python create_latency_estimator.py \
        --accum-latency MNV2-CIFAR10-sampled-nnps-accum-latency \
        --sampled-latency MNV2-CIFAR10-sampled-latency

One can find the estimator.py with the scale and bias being encoded,
which is used for the estimation of a latency.


