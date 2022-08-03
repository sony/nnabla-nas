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

import os
from collections import OrderedDict
import nnabla as nn
from nnabla.utils.save import save
from .parameter import Parameter
from hydra import utils


class Module(object):
    r"""Module base for all nnabla neural network modules.

    Your models should also subclass this class. Modules can also contain
    other Modules, allowing to nest them in a tree structure.
    """
    def __init__(self, name=''):
        self._name = name
        if os.environ.get('NNABLA_NAS_MIXEDOP_FAST_MODE') is not None:
            self._call_create = self.call
            self.call = self._call_cached
            self._train_output = None
            self._infer_output = None

    @property
    def name(self):
        r"""
        The name of the module.

        Returns:
            string: the name of the module
        """
        return self._name

    @property
    def modules(self):
        r"""Return an `OrderedDict` containing immediate modules."""
        if '_modules' not in self.__dict__:
            self.__dict__['_modules'] = OrderedDict()
        return self._modules

    @property
    def parameters(self):
        r"""Return an `OrderedDict` containing immediate parameters."""
        if '_parameters' not in self.__dict__:
            self.__dict__['_parameters'] = OrderedDict()
        return self._parameters

    @property
    def training(self):
        r"""The training mode of module."""
        if '_training' not in self.__dict__:
            self.__dict__['_training'] = True
        return self._training

    @training.setter
    def training(self, mode):
        self.__dict__['_training'] = mode
        for _, m in self.modules.items():
            m.training = mode

    @property
    def need_grad(self):
        r"""Whether the module needs gradient."""
        if '_need_grad' not in self.__dict__:
            self.__dict__['_need_grad'] = True
        return self._need_grad

    @need_grad.setter
    def need_grad(self, mode):
        self.__dict__['_need_grad'] = mode
        for _, m in self.modules.items():
            m.need_grad = mode

    @property
    def is_active(self):
        r"""Whether the module was called."""
        if '_is_active' not in self.__dict__:
            self.__dict__['_is_active'] = True
        return self._is_active

    @is_active.setter
    def is_active(self, mode):
        self.__dict__['_is_active'] = mode

    @property
    def input_shapes(self):
        r"""Return a list of input shapes used during `call` function."""
        if '_input_shapes' not in self.__dict__:
            self.__dict__['_input_shapes'] = list()
        return self._input_shapes

    @input_shapes.setter
    def input_shapes(self, v):
        setattr(self, '_input_shapes', v)

    def _get_need_grad_state(self):
        from nnabla import parameter
        # TODO: Ideally want to have get_current_no_grad_state() in parameter module?
        no_grad = parameter.current_no_grad
        if no_grad:
            return False
        return self.need_grad

    def __getattr__(self, name):
        if name in self.modules:
            return self.modules[name]
        if name in self.parameters:
            need_grad_state = self._get_need_grad_state()
            p = self.parameters[name]
            if not need_grad_state and p.need_grad:
                return p.get_unlinked_variable(need_grad=need_grad_state)
            return p
        return object.__getattr__(self, name)

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                d.pop(name, None)
        remove_from(self.__dict__, self.modules, self.parameters)
        if isinstance(value, Module):
            self.modules[name] = value
        elif isinstance(value, Parameter):
            self.parameters[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self.parameters:
            del self.parameters[name]
        elif name in self.modules:
            del self.modules[name]
        else:
            object.__delattr__(self, name)

    def apply(self, memo=None, **kargs):
        r"""Helper for setting property recursively, then returns self."""
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            for key, value in kargs.items():
                setattr(self, key, value)
            for module in self.modules.values():
                module.apply(memo, **kargs)
        return self

    def get_modules(self, prefix='', memo=None):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            prefix (str, optional): Additional prefix to name modules.
                Defaults to ''.
            memo (dict, optional): Memorize all parsed modules.
                Defaults to None.

        Yields:
            (str, Module): a submodule.
        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self.modules.items():
                submodule_prefix = prefix + ('/' if prefix else '') + name
                for m in module.get_modules(submodule_prefix, memo):
                    yield m

    def get_parameters(self, grad_only=False):
        r"""Return an `OrderedDict` containing all parameters in the module.

        Args:
            grad_only (bool, optional): If need_grad=True is required.
                Defaults to False.

        Returns:
            OrderedDict: A dictionary containing parameters of module.
        """
        params = OrderedDict()
        for prefix, module in self.get_modules():
            if grad_only and not module.need_grad:
                continue
            for name, p in module.parameters.items():
                if grad_only and not p.need_grad:
                    continue
                key = prefix + ('/' if prefix else '') + name
                params[key] = p
        return params

    def set_parameters(self, params, raise_if_missing=False):
        r"""Set parameters for the module.

        Args:
            params (OrderedDict): The parameters which will be loaded.
            raise_if_missing (bool, optional): Raise exception if some
                parameters are missing. Defaults to `False`.

        Raises:
            ValueError: Parameters are not found.
        """
        for prefix, module in self.get_modules():
            for name, p in module.parameters.items():
                key = prefix + ('/' if prefix else '') + name
                if key in params:
                    p.d = params[key].d.copy()
                    nn.logger.info(f'`{key}` loaded.')
                elif raise_if_missing:
                    raise ValueError(
                        f'A child module {name} cannot be found in '
                        '{this}. This error is raised because '
                        '`raise_if_missing` is specified '
                        'as True. Please turn off if you allow it.')

    def save_parameters(self, path, params=None, grad_only=False):
        r"""Saves the parameters to a file.

        Args:
            path (str): Path to file.
            params (OrderedDict, optional): An `OrderedDict` containing
                parameters. If params is `None`, then the current parameters
                will be saved.
            grad_only (bool, optional): If need_grad=True is required for
                parameters which will be saved. Defaults to False.
        """
        params = params or self.get_parameters(grad_only)
        nn.save_parameters(path, params)

    def load_parameters(self, path, raise_if_missing=False):
        r"""Loads parameters from a file with the specified format.

        Args:
            path (str): The path to file.
            raise_if_missing (bool, optional): Raise exception if some
                parameters are missing. Defaults to `False`.
        """
        with nn.parameter_scope('', OrderedDict()):
            load_path = os.path.realpath(os.path.join(utils.get_original_cwd(), path))  # because hydra changes
            nn.load_parameters(load_path)                                               # the working directory
            params = nn.get_parameters(grad_only=False)
        self.set_parameters(params, raise_if_missing=raise_if_missing)

    @property
    def modules_to_profile(self):
        r"""Returns a list with the modules that will be profiled when the
        Profiler/Estimator functions are called. All other modules in the
        network will not be profiled.
        """
        raise NotImplementedError

    def get_latency(self, estimator, active_only=True):
        """
        Function to use to calc latency
        This function needs to work based on the graph
        Parameters:
            estimator: a graph-based estimator
            active_only: get latency of active modules only
        Returns:
            latencies: list of all latencies of each module
            accum_lat: total sum of latencies of all modules
        """
        accum_lat = 0
        latencies = {}
        for mi in self.get_net_modules(active_only=active_only):
            if type(mi) in self.modules_to_profile:
                inp = [nn.Variable((1,)+si[1:]) for si in mi.input_shapes]
                out = mi.call(*inp)
                latencies[mi.name] = estimator.predict(out)
                accum_lat += latencies[mi.name]
        return latencies, accum_lat

    def get_latency_by_mod(self, estimator, active_only=True):
        """
        *** Note: This function is deprecated. Use get_latency() ***
        Function to use to calc latency
        This function needs to work based on the module
        Parameters:
            estimator: a module-based estimator
            active_only: get latency of active modules only
        Returns:
            latencies: list of all latencies of each module
            accum_lat: total sum of latencies of all modules
        """
        accum_lat = 0
        latencies = {}
        for mi in self.get_net_modules(active_only=active_only):
            if type(mi) in self.modules_to_profile:
                latencies[mi.name] = estimator.predict(mi)
                accum_lat += latencies[mi.name]
        return latencies, accum_lat

    def save_net_nnp(self, path, inp, out, calc_latency=False,
                     func_real_latency=None, func_accum_latency=None,
                     save_params=None):
        """
        Saves whole net as one nnp Calc whole net (real) latency (using
        e.g.Nnabla's [Profiler]) Calculate also layer-based latency The modules
        are discovered using the nnabla graph of the whole net The latency is
        then calculated based on each individual module's nnabla graph (e.g.
        [LatencyGraphEstimator])

        Args:
            path
            inp: input of the created network
            out: output of the created network
            calc_latency: flag for calc latency
            func_real_latency: function to use to calc actual latency
            func_accum_latency: function to use to calc accum. latency,
                    this is, dissecting the network layer by layer
                    using the graph of the network, calculate the
                    latency for each layer and add up all these results.
        """
        batch_size = inp.shape[0]

        name = self.name if (hasattr(self, 'name')
                             and self.name) else 'results'

        filename = os.path.join(path, name + '.nnp')
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        name_for_nnp = self.name if (
            hasattr(self, 'name') and self.name) else 'empty'
        contents = {'networks': [{'name': name_for_nnp,
                                  'batch_size': batch_size,
                                  'outputs': {"y'": out},
                                  'names': {'x': inp}}],
                    'executors': [{'name': 'runtime',
                                   'network': name_for_nnp,
                                   'data': ['x'],
                                   'output': ["y'"]}]}
        if save_params and 'no_image_normalization' in save_params:
            contents['executors'][0]['no_image_normalization'] = save_params['no_image_normalization']

        save(filename, contents, variable_batch_size=False)

        if calc_latency:
            acc_latency = func_accum_latency.get_estimation(out)
            filename = path + name + '.acclat'
            with open(filename, 'w') as f:
                print(acc_latency.__str__(), file=f)

            func_real_latency.run()
            real_latency = float(func_real_latency.result['forward_all'])
            filename = path + name + '.realat'
            with open(filename, 'w') as f:
                print(real_latency.__str__(), file=f)
            return real_latency, acc_latency
        else:
            return 0.0, 0.0

    def save_modules_nnp(self, path, active_only=False,
                         calc_latency=False,
                         func_latency=None
                         ):
        """
        Saves all modules of the network as individual nnp files, using folder
        structure given by name convention.  The modules are extracted going
        over the module list, not over the graph structure.  The latency is
        then calculated based on each individual module's nnabla graph (e.g.
        [LatencyGraphEstimator])

        Args:
            path
            active_only: if True, only active modules are saved
            calc_latency: flag for calc latency
            func_latency: function to use to calc latency of
                          each of the extracted modules
                          This function needs to work based on the graph
        """
        accum_lat = 0.0
        mods = self.get_net_modules(active_only=active_only)
        for mi in mods:
            if type(mi) in self.modules_to_profile:
                if len(mi.input_shapes) == 0:
                    continue
                pass

                inp = [nn.Variable((1,)+si[1:]) for si in mi.input_shapes]
                out = mi.call(*inp)

                filename = path + mi.name + '.nnp'
                pathname = os.path.dirname(filename)
                upper_pathname = os.path.dirname(pathname)
                if not os.path.exists(upper_pathname):
                    os.mkdir(upper_pathname)
                if not os.path.exists(pathname):
                    os.mkdir(pathname)

                d_dict = {str(i): inpi for i, inpi in enumerate(inp)}
                d_keys = [str(i) for i, inpi in enumerate(inp)]

                name_for_nnp = mi.name if (mi.name != '') else 'empty'
                contents = {'networks': [{'name': name_for_nnp,
                                          'batch_size': 1,
                                          'outputs': {'out': out},
                                          'names': d_dict}],
                            'executors': [{'name': 'runtime',
                                           'network': name_for_nnp,
                                           'data': d_keys,
                                           'output': ['out']}]}

                if hasattr(mi, '_scope_name'):
                    with nn.parameter_scope(mi._scope_name):
                        save(filename, contents, variable_batch_size=False)
                else:
                    save(filename, contents, variable_batch_size=False)

                if calc_latency:
                    latency = func_latency.get_estimation(out)
                    filename = path + mi.name + '.acclat'
                    with open(filename, 'w') as f:
                        print(latency.__str__(), file=f)
                    accum_lat += latency
        return accum_lat

    def save_modules_nnp_by_mod(self, path, active_only=False,
                                calc_latency=False,
                                func_latency=None,
                                ):
        """
        *** Note: This function is deprecated. Use save_modules_nnp() ***
        Saves all modules of the network as individual nnp files,
        using folder structure given by name convention.
        The modules are extracted going over the module list, not
        over the graph structure.
        The latency is then calculated using the module themselves
        (e.g. [LatencyEstimator])

        Args:
            path
            active_only: if True, only active modules are saved
            calc_latency: flag for calc latency
            func_latency: function to use to calc latency of
                          each of the extracted modules
                          This function needs to work based on the modules
        """
        accum_lat = 0.0
        mods = self.get_net_modules(active_only=active_only)
        for mi in mods:
            if type(mi) in self.modules_to_profile:
                if len(mi.input_shapes) == 0:
                    continue
                pass

                inp = [nn.Variable((1,)+si[1:]) for si in mi.input_shapes]
                out = mi.call(*inp)

                filename = path + mi.name + '.nnp'
                pathname = os.path.dirname(filename)
                upper_pathname = os.path.dirname(pathname)
                if not os.path.exists(upper_pathname):
                    os.mkdir(upper_pathname)
                if not os.path.exists(pathname):
                    os.mkdir(pathname)

                d_dict = {str(i): inpi for i, inpi in enumerate(inp)}
                d_keys = [str(i) for i, inpi in enumerate(inp)]

                name_for_nnp = mi.name if (mi.name != '') else 'empty'
                contents = {'networks': [{'name': name_for_nnp,
                                          'batch_size': 1,
                                          'outputs': {'out': out},
                                          'names': d_dict}],
                            'executors': [{'name': 'runtime',
                                           'network': name_for_nnp,
                                           'data': d_keys,
                                           'output': ['out']}]}

                if hasattr(mi, '_scope_name'):
                    with nn.parameter_scope(mi._scope_name):
                        save(filename, contents, variable_batch_size=False)
                else:
                    save(filename, contents, variable_batch_size=False)

                if calc_latency:
                    latency = func_latency.get_estimation(mi)
                    filename = path + mi.name + '.acclat'
                    with open(filename, 'w') as f:
                        print(latency.__str__(), file=f)
                    accum_lat += latency
        return accum_lat

    def calc_latency_all_modules(self, path, graph, func_latency=None):
        """
        Calculate the latency for each of the modules in a graph.
        The modules are extracted using the graph structure information.
        The latency is then calculated based on each individual module's
        nnabla graph.
        It also saves the accumulated latency of all modules.

        Args:
            path
            graph:
            func_latency: function to use to calc latency of
                          each of the modules
                          This function needs to work based on the graph
        """
        import nnabla.function as Function
        from nnabla_nas.utils.estimator.latency import Profiler
        from nnabla.context import get_current_context
        from nnabla.logger import logger

        func_latency._visitor.reset()
        graph.visit(func_latency._visitor)
        total_latency = 0.0
        idx = 0
        for func in func_latency._visitor._functions:
            args = [func.info.type_name] + \
                   [str(inp.shape) for inp in func.inputs] + \
                   [str(func.info.args)]
            key = '-'.join(args)

            ff = getattr(Function, func.info.type_name)(get_current_context(),
                                                        **func.info.args)

            if key not in func_latency.memo:
                try:  # run profiler
                    nnabla_vars = [nn.Variable(inp.shape,
                                   need_grad=inp.need_grad)
                                   for inp in func.inputs]
                    runner = Profiler(
                        ff(*nnabla_vars),
                        device_id=func_latency._device_id,
                        ext_name=func_latency._ext_name,
                        n_run=func_latency._n_run,
                        outlier=func_latency._outlier,
                        max_measure_execution_time=func_latency._max_measure_execution_time,  # noqa: E501
                        time_scale=func_latency._time_scale,
                        n_warmup=func_latency._n_warmup
                    )
                    runner.run()
                    latency = float(runner.result['forward_all'])

                except Exception as err:
                    latency = 0.0
                    logger.warning(f'Latency calculation failed: {key}')
                    logger.warning(str(err))

                func_latency.memo[key] = latency
            else:
                latency = func_latency.memo[key]

            total_latency += latency

            # save latency of this layer (name: id_XXX_{key}.acclat)
            filename = path + '/id_' + str(idx) + '_' + key + '.acclat'
            pathname = os.path.dirname(filename)
            upper_pathname = os.path.dirname(pathname)
            if not os.path.exists(upper_pathname):
                os.mkdir(upper_pathname)
            if not os.path.exists(pathname):
                os.mkdir(pathname)
            idx += 1
            with open(filename, 'w') as f:
                print(latency.__str__(), file=f)

        # save accum latency of all layers
        filename = path + '.acclat'
        with open(filename, 'w') as f:
            print(total_latency.__str__(), file=f)

        return total_latency

    def convert_npp_to_onnx(self, path, opset='opset_11'):
        """
        Finds all nnp files in the given path and its subfolders and converts
        them to ONNX For this to run smoothly, nnabla_cli must be installed and
        added to your python path.

        Args:
            path
            opset

        The actual bash shell command used is::

            > find <DIR> -name '*.nnp' -exec echo echo {} \|
              awk -F \\. \'\{print \"nnabla_cli convert -b 1 -d opset_11 \"\$0\" \"\$1\"\.\"\$2\"\.onnx\"\}\' \; | sh | sh

        which, for each file found with find, outputs the following::

            > echo <FILE>.nnp | awk -F \. '{print "nnabla_cli convert -b 1 -d opset_11 "$0" "$1"."$2".onnx"}'  # noqa: E501,W605

        which, for each file, generates the final conversion command::

            > nnabla_cli convert -b 1 -d opset_11 <FILE>.nnp <FILE>.nnp.onnx

        """
        os.system('find ' + path + ' -name "*.nnp" -exec echo echo {} \|'              # noqa: E501,W605
                  ' awk -F \\. \\\'{print \\\"nnabla_cli convert -b 1 -d ' + opset +   # noqa: E501,W605
                  ' \\\"\$0\\\" \\\"\$1\\\"\.\\\"\$2\\\"\.onnx\\\"}\\\' \; | sh | sh'  # noqa: E501,W605
                  )

    def extra_format(self):
        r"""Set the submodule representation format.
        """
        return '.{}'

    def extra_repr(self):
        r"""Set the extra representation for the module."""
        return ''

    def __str__(self):
        r"""Return str representtation of the module."""
        main_str = f'{self.__class__.__name__}(' + self.extra_repr()
        sub_str = ''
        for key, module in self.modules.items():
            m_repr = str(module).split('\n')
            head = [self.extra_format().format(key) + ': ' + m_repr.pop(0)]
            tail = [m_repr.pop()] if len(m_repr) else []
            m_repr = [' '*2 + line for line in (head + m_repr + tail)]
            sub_str += '\n' + '\n'.join(m_repr)
        main_str += sub_str + ('\n' if sub_str else '') + ')'
        return main_str

    def __call__(self, *args, **kwargs):
        self.input_shapes = [x.shape for x in args]
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        r"""Implement the call of module. Inputs should only be Variables."""
        raise NotImplementedError

    def _call_cached(self, *args, **kwargs):
        if self.training:
            if self._train_output is None:
                self._train_output = self._call_create(*args, **kwargs)
            return self._train_output
        else:
            if self._infer_output is None:
                self._infer_output = self._call_create(*args, **kwargs)
            return self._infer_output
