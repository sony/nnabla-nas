# Latency Measurements for NNabla Extension

## Prepare NNabla

Install NNabla, 
```
pip install -U nnabla nnabla-ext-cuda
```

or use the docker [image](https://hub.docker.com/r/nnabla/nnabla-ext-cuda/tags)

```
docker pull nnabla/nnabla-ext-cuda:latest
```

and use that image for latency measurement.

## NAS repository

```bash
git clone https://gitlab.stc.eu.sony.com/bacnguyencong/nnabla_nas
export PYTHONPATH=${PWD}/nnabla_nas:$PYTHONPATH
```

**NOTE** 
The following example is based on MobileNetV-2 Search Space and CIFAR-10 dataset. 
Change the settings if you want to change a search space and dataset.


## Measure latency and Create Latency Table

Meaure latency, then create the latency table, 

```bash
python create_latency_table.py \
    --search-net-config ../../../../examples/mobilenet_cifar10_search.json \
    --table-name MNV2-CIFAR10-space-latency \
    --n-run 100 \
    --time-scale m \
```

Table looks like

In general, the format is the following.

```
<ModuleUID>,<Runtime>,<Value>
```

- ModuleUID: Module Unique ID to measure latency
- Runtime: Runtime like cpu:float, cudnn:flot, cuda:half
- Value: latency (**milli** second)


```bash
python create_latency_estimator.py \
    --search-net-config ../../../../examples/mobilenet_cifar10_search.json \
    --latency-table-json MNV2-CIFAR10-space-latency.json \
    --n-run 100 \
    --time-scale m \
    --num-trials 100
```

You can find the estimator.py with the scale and bias being encoded, 
which is used for the estimation of a latency.

