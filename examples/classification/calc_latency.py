import nnabla as nn
from nnabla_nas.contrib.classification.zoph import SearchNet as Zoph
from nnabla_nas.contrib.classification.random_wired import TrainNet as Rdw
from nnabla_nas.contrib.classification.mobilenet import SearchNet as MobileNet

from nnabla_nas.utils.estimator.latency import LatencyGraphEstimator, Profiler
from nnabla.ext_utils import get_extension_context

all_networks = {'zoph': Zoph, 'randomly_wired': Rdw, 'mobilenet_v2': MobileNet}

# Which example network to run? OPTIONS:
# 'zoph': a ZOPH network
# 'randomly_wired': a RANDOM WIRED network
# 'mobilenet_v2': a MOBILENETv2 network
network_to_run = 'mobilenet_v2'
print(' ** Running example for ' + network_to_run + ' network **')

# Where to run the latency estimation ?
ext_name = 'cudnn'  # OPTIONS: 'cpu', 'cuda', 'cudnn'
device_id = 0
ctx = get_extension_context(ext_name=ext_name, device_id=device_id)
nn.set_default_context(ctx)

# Params for the latency estimation
max_measure_execution_time = 500
time_scale = "m"
n_warmup = 10
n_run = 100
# Portion of outliers which will be removed from the statistical results
outlier = 0.05

model = all_networks[network_to_run]()
out = model(nn.Variable((1, 3, 32, 32)))

for i in range(10):
    estimator = LatencyGraphEstimator(
        device_id=device_id, ext_name=ext_name,
        outlier=outlier,
        time_scale=time_scale,
        n_warmup=n_warmup,
        max_measure_execution_time=max_measure_execution_time,
        n_run=n_run
    )

    runner = Profiler(
        out,
        device_id=device_id, ext_name=ext_name,
        outlier=outlier,
        time_scale=time_scale,
        n_warmup=n_warmup,
        max_measure_execution_time=max_measure_execution_time,
        n_run=n_run
    )

    print('- running LatencyGraphEstimator ...')
    lat2 = estimator.get_estimation(out)

    print('- running Profiler ...')
    runner.run()
    latency = float(runner.result['forward_all'])

    print(f'X-> run={i}\t real_latency(Profiler)=' +
          f'{latency:.5f}ms\t accum_latency(LatencyGraphEstimation)=' +
          f'{lat2:.5f}ms\t n_functions={len(estimator._visitor._functions)}'
          )
