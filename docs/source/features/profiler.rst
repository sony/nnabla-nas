Profiler
--------

NNabla NAS has the offline profiler. Profiler measures the latency for any modules; a single module and/or the network made of modules. The offline profiling is necessary since if one wants to measure latency of a module on-the-fly during a search phase, which could be a bottleneck in a training system, and addtionally one has to attach and incoorporate a target device to a training system.

Currently, this profiler assumes the Proxyless NAS algorithm, and the following devices and runtimes are tested.


:ref:`NNabla <profiler-nnabla>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Intel CPU (cpu:float)
- NVIDIA GPU (cudnn:float)


:ref:`SNPE <profiler-snpe>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- CPU
- GPU
- GPU_FP16
- DSP

:ref:`TensorRT <profiler-tensorrt>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

COMMING SOON.



