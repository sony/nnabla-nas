import nnabla as nn
from nnabla_nas.contrib.classification.zoph import SearchNet as Zoph
from nnabla_nas.contrib.classification.random_wired import TrainNet as Rdn
from nnabla_nas.contrib.classification.mobilenet import SearchNet as MobileNet

from nnabla_nas.utils.estimator.latency import LatencyGraphEstimator, Profiler
from nnabla.ext_utils import get_extension_context

# Which example network to run? OPTIONS:
# 0: a ZOPH network
# 1: a RANDOM WIRED network
# 2: a MOBILENET network
network_to_run = 0

# Where to run the latency estimation ?
ext_name = 'cudnn'  # OPTIONS: 'cpu', 'cuda', 'cudnn'
device_id = 0
ctx = get_extension_context(ext_name=ext_name, device_id=device_id)
nn.set_default_context(ctx)

# Params for the latency estimation
outlier = 0.05
max_measure_execution_time = 500
time_scale = "m"
n_warmup = 10
n_run = 100

if network_to_run == 0:
    print('** Example for ZOPH Network ***')
    model = Zoph()
elif network_to_run == 1:
    print('** Example for RANDOM WIRED Network ***')
    model = Rdn()
else:
    print('** Example for MOBILENET Network ***')
    model = MobileNet()
pass
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

    runner = Profiler(out,
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
