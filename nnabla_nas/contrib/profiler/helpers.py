import nnabla as nn
import nnabla.utils.save as save

import nnabla_nas

from subprocess import Popen, PIPE, STDOUT


def force_list(inp):
    if isinstance(inp, (tuple)):
        return list(inp)
    if not isinstance(inp, (list)):
        return [inp]
    else:
        return inp


def uid(module):
    # Friendly with `bash -c`
    key = str(module) + str([shape for shape in module.input_shapes])
    key = key.replace(" ", "")
    key = key.replace("(", "[")
    key = key.replace(")", "]")
    return key


def get_unique_modules(net,
                       skip_modules=(nnabla_nas.module.identity.Identity,
                                     nnabla_nas.module.dropout.Dropout
                                     )):
    mods = [mi for _, mi in net.get_modules()]
    unique_mods = {}
    for mi in mods:
        if len(mi.modules) != 0 or not mi.need_grad or len(mi.input_shapes) == 0:
            continue
        if isinstance(mi, skip_modules):
            continue
        unique_mods[uid(mi)] = mi
    return unique_mods


def get_sampled_modules(net, skip_modules=(nnabla_nas.module.identity.Identity,
                                           nnabla_nas.module.dropout.Dropout)):
    modules = [m for _, m in net.get_modules()
               if len(m.modules) == 0 and m.need_grad
               and not isinstance(m, skip_modules)]
    return modules


def get_search_net(name, num_classes, mode):
    if name == "mbn":
        from nnabla_nas.contrib.mbn import SearchNet
        net = SearchNet(num_classes=num_classes, mode=mode)
    if name == "darts":
        from nnabla_nas.contrib.darts import SearchNet
        net = SearchNet(num_classes=num_classes, mode=mode)
    return net.apply(training=False)


class CreateParameters(object):

    def __init__(self):
        self._idx = 0

    def update_idx(func):
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self._idx += 1
        return wrapper

    def __call__(self, func):
        if func.info.type_name == "Convolution":
            self.create_conv_parameters(func)
        elif func.info.type_name == "Deconvolution":
            self.create_deconv_parameters(func)
        elif func.info.type_name == "Affine":
            self.create_affine_parameters(func)
        elif func.info.type_name == "BatchNormalization":
            self.create_batchnorm_parameters(func)

    @update_idx
    def create_conv_parameters(self, func):
        nn.parameter.set_parameter("conv_weights-{}".format(self._idx), func.inputs[1])
        if len(func.inputs) > 2:
            nn.parameter.set_parameter("conv_bias-{}".format(self._idx), func.inputs[2])

    @update_idx
    def create_deconv_parameters(self, func):
        nn.parameter.set_parameter("deconv_weights-{}".format(self._idx), func.inputs[1])
        if len(func.inputs) > 2:
            nn.parameter.set_parameter("deconv_bias-{}".format(self._idx), func.inputs[2])

    @update_idx
    def create_affine_parameters(self, func):
        nn.parameter.set_parameter("affine_weights-{}".format(self._idx), func.inputs[1])
        if len(func.inputs) > 2:
            nn.parameter.set_parameter("affine_bias-{}".format(self._idx), func.inputs[2])

    @update_idx
    def create_batchnorm_parameters(self, func):
        nn.parameter.set_parameter("bn_beta-{}".format(self._idx), func.inputs[1])
        nn.parameter.set_parameter("bn_gamma-{}".format(self._idx), func.inputs[2])
        nn.parameter.set_parameter("bn_mu-{}".format(self._idx), func.inputs[3])
        nn.parameter.set_parameter("bn_sigma-{}".format(self._idx), func.inputs[4])


def create_parameters(out):
    ftor = CreateParameters()
    out.visit(ftor)


def nnp_save(path, mod_name, inp, out):
    names = {}
    data = []
    inp = force_list(inp)
    for i, inp_ in enumerate(inp):
        name = "x{}".format(i)
        names[name] = inp_
        data.append(name)
    runtime_contents = {
        'networks': [
            {'name': '{}'.format(mod_name),
             'batch_size': 1,
             'outputs': {'y': out},
             'names': names}],
        'executors': [
            {'name': 'Runtime',
             'network': '{}'.format(mod_name),
             'data': data,
             'output': ['y']}]}

    print("Saving {} ...".format(path))
    save.save(path, runtime_contents, variable_batch_size=False)


def run_command(cmdline):
    p = Popen(cmdline, shell=True, stdout=PIPE, stderr=STDOUT, close_fds=True)
    for ln in p.stdout:
        out_line = ln.decode('utf-8').strip()
        print(out_line)
    ret_code = p.wait()
    if ret_code != 0:
        raise ValueError("{}".format(cmdline))
