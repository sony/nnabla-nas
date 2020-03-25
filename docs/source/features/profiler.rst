Profiler
--------

NNabla NAS has an offline profiler. Profiler measures the latency for any modules; a single module and/or the network made of modules. Offline profiling is necessary since if one wants to measure the latency of a module on-the-fly during a search phase, which could be a bottleneck in a training system, and additionally one has to attach and incorporate a target device to a training system.

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

COMING SOON.