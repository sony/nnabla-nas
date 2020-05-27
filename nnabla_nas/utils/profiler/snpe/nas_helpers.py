# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from subprocess import Popen, PIPE, STDOUT


class BatchNormFoldingChecker():
    def __init__(self, mod_name):
        self._conv_bn = False
        self._mod_name = mod_name

    def uid_bn(self, bn_func):
        inp = bn_func.inputs[0]
        ishape = inp.shape
        n_features = ishape[bn_func.info.args["axes"][0]]
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


def run_command(cmdline):
    p = Popen(cmdline, shell=True, stdout=PIPE, stderr=STDOUT, close_fds=True)
    for ln in p.stdout:
        out_line = ln.decode('utf-8').strip()
        print(out_line)
    ret_code = p.wait()
    if ret_code != 0:
        raise ValueError("{}".format(cmdline))
