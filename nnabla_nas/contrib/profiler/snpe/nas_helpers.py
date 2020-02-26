import json
from subprocess import Popen, PIPE, STDOUT


class BatchNormFoldingChecker():
    def __init__(self, mod_name):
        self._conv_bn = False
        self._mod_name = mod_name

    def uid_bn(self, bn_func):
        inp = bn_func.inputs[0]
        ishape = inp.shape
        n_features = ishape[bn_func.info.args["axes"][0]]
        eps = bn_func.info.args["eps"]
        decay_rate = bn_func.info.args["decay_rate"]
        s_ishape = ",".join([str(s) for s in ishape])
        ## uid = "BatchNormalization[n_features={},fix_parameters=False,eps={},decay_rate={}][[{}]]"\
        # .format(n_features, eps, decay_rate, ",".join([str(s) for s in ishape]))
        ## uid = f"BatchNormalization[n_features={n_features},fix_parameters=False,eps={eps},decay_rate={decay_rate}][[{s_ishape}]]"
        uid = "BatchNormalization[n_features={},fix_parameters=False,eps=1e-05,decay_rate=0.9][[{}]]"\
            .format(n_features, ",".join([str(s) for s in ishape]))

        return uid

    def __call__(self, f):
        if f.info.type_name != "BatchNormalization":
            return
        if f.inputs[0].parent.info.type_name not in \
                ["Convolution", "Deconvolution", "Affine"]:
            return

        if self._mod_name == self.uid_bn(f):
            self._conv_bn = True

    def result(self):
        return self._conv_bn


def check_bn_fold(mod_name, out):
    if "BatchNormalization" not in mod_name:
        return False
    callback = BatchNormFoldingChecker(mod_name)
    out.visit(callback)
    return callback.result()


# class SnpeLatencyEstimator(Estimator):

# def __init__(self, latency_table_json, runtime):
# """
# Args:
# latency_table_json (str): Path to the latency table.
# runtime (str): Key to the runtime in ["CPU", "GPU", "GPU_FP16", "DSP"].
# """
# with open(latency_table_json) as fp:
##             self._latency_table = json.load(fp)
##         self._runtime = runtime

# self._skip_modules = (nnabla_nas.module.identity.Identity,
# nnabla_nas.module.dropout.Dropout)

# TODO: remove BN of `Conv -> BN`
# def predict(self, module):
# if isinstance(module, self._skip_modules):
# return 0.0
# return self._latency_table[uid(module)][self._runtime]["Layers Time"]["Avg_Time"]
# return self._latency_table[uid(module)][self._runtime]["Forward Propagate"]["Avg_Time"]


def run_command(cmdline):
    p = Popen(cmdline, shell=True, stdout=PIPE, stderr=STDOUT, close_fds=True)
    for ln in p.stdout:
        out_line = ln.decode('utf-8').strip()
        print(out_line)
    ret_code = p.wait()
    if ret_code != 0:
        raise ValueError("{}".format(cmdline))
