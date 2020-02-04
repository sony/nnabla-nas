import numpy as np
import nnabla as nn
import nnabla.functions as F
from collections import OrderedDict
from nnabla.initializer import ConstantInitializer
import nnabla_nas.module as mo
import nnabla_nas.module.static.static_module as smo
from nnabla_nas.module.parameter import Parameter

#---------------------Definition of the candidate convolutions for the zoph search space-------------------------------
class SepConv(smo.SepConv):
    def __init__(self, name, parent, out_channels, kernel, dilation, eval_prob=None):

        if dilation is None:
            pad = tuple([ki//2 for ki in kernel])
        else:
            pad = tuple([(ki//2)*di for ki,di in zip(kernel, dilation)])

        smo.SepConv.__init__(self, name=name, parent=parent, in_channels=parent.shape[1],
                             out_channels=parent.shape[1],
                             kernel=kernel, pad=pad,
                             dilation=dilation, with_bias=False,
                             eval_prob=eval_prob)

        self.bn = mo.BatchNormalization(n_features=self._out_channels, n_dims=4)
        self.relu = mo.ReLU()

    def _value_function(self, input):
        return self.relu(self.bn(smo.SepConv._value_function(self, smo.SepConv._value_function(self, input))))

class SepConv3x3(SepConv):
    def __init__(self, name, parent, channels, eval_prob=None):
        SepConv.__init__(self, name, parent, channels, (3,3), None, eval_prob=eval_prob)

class SepConv5x5(SepConv):
    def __init__(self, name, parent, channels, eval_prob=None):
        SepConv.__init__(self, name, parent, channels, (5,5), None, eval_prob=eval_prob)

class DilSepConv3x3(SepConv):
    def __init__(self, name, parent, channels, eval_prob=None):
        SepConv.__init__(self, name, parent, channels, (3,3), (2,2), eval_prob=eval_prob)

class DilSepConv5x5(SepConv):
    def __init__(self, name, parent, channels, eval_prob=None):
        SepConv.__init__(self, name, parent, channels, (5,5), (2,2), eval_prob=eval_prob)


#---------------------Definition of the candidate pooling operations for the zoph search space-------------------------------
class MaxPool3x3(smo.MaxPool):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        smo.MaxPool.__init__(self, name, parent, kernel=(3,3), stride=(1,1), pad=(1,1))
        self.bn = mo.BatchNormalization(n_features=self.parent.shape[1], n_dims=4)
        self.relu= mo.ReLU()

    def _value_function(self, input, *args, **kwargs):
        return self.relu(self.bn(smo.MaxPool._value_function(self, input)))

class AveragePool3x3(smo.AvgPool):
    def __init__(self, name, parent, eval_prob=None, *args, **kwargs):
        smo.AvgPool.__init__(self, name, parent, kernel=(3,3), stride=(1,1), pad=(1,1), eval_prob=eval_prob)
        self.bn = mo.BatchNormalization(n_features=self.parent.shape[1], n_dims=4)
        self.relu= mo.ReLU()

    def _value_function(self, input):
        return self.relu(self.bn(smo.AvgPool._value_function(self, input)))

#---------------------List of candidate layers for the zoph search space-------------------------------
ZOPH_CANDIDATES = [SepConv3x3,
                   SepConv5x5,
                   DilSepConv3x3,
                   DilSepConv5x5,
                   MaxPool3x3,
                   AveragePool3x3,
                   smo.Identity,
                   smo.Zero]

#-----------------------------------The definition of a Zoph cell--------------------------------------
class ZophBlock(smo.Graph):
    def __init__(self, name, parents, candidates, channels, join_parameters=None):
        self._candidates = candidates
        self._channels = channels
        if join_parameters is None:
            self._join_parameters = Parameter(shape=(len(candidates),))
        else:
            self._join_parameters = join_parameters
        smo.Graph.__init__(self, name, parents)

        join_prob = F.softmax(self._join_parameters)

        #add an input concatenation
        input_con = smo.Merging(name='{}/input_con'.format(self.name),
                                parents=self.parent,
                                mode='concat',
                                axis=1,
                                eval_prob=F.sum(join_prob[:-1]))
        self.append(input_con)
        input_conv = smo.Conv(name='{}/input_conv'.format(self.name), parent=input_con, in_channels=input_con.shape[1],
                              out_channels=self._channels, kernel=(1,1),
                                eval_prob=F.sum(join_prob[:-1]))
        self.append(input_conv)

        for i,ci in enumerate(self._candidates):
            self.append(ci(name='{}/candidate_{}'.format(self.name, i),
                           parent=input_conv,
                           channels=self._channels,
                           eval_prob=join_prob[i]))
        self.append(smo.Join(name='{}/join'.format(self.name), parents=self[2:], join_parameters=self._join_parameters))

class ZophCell(smo.Graph):
    def __init__(self, name, parents, candidates, channels, n_modules=3, reducing=False, join_parameters=[None]*3):
        self._candidates = candidates
        self._channels = channels
        self._n_modules = n_modules
        self._reducing = reducing
        self._join_parameters = join_parameters
        smo.Graph.__init__(self, name, parents)

        #match the input dimensions
        shapes = [(list(ii.shape) + 4 * [1])[:4] for ii in self.parent]
        min_shape = np.min(np.array(shapes), axis=0)
        self._shape_adaptation = {i: np.array(si[2:]) / min_shape[2:] for i,si in enumerate(shapes) if tuple(si[2:]) != tuple(min_shape[2:])}

        #perform the input channel projection, using pointwise convolutions
        projected_inputs = []
        for i,ii in enumerate(self.parent):
            self.append(smo.Conv(name='{}/input_conv_{}'.format(self.name, i),
                                       parent=ii, in_channels=ii.shape[1],
                                       out_channels=self._channels,
                                       kernel=(1,1), with_bias=False))
            projected_inputs.append(self[-1])

        #perform shape adaptation, using pooling, if needed
        for i,pii in enumerate(projected_inputs):
            if i in self._shape_adaptation:
                self.append(smo.MaxPool(name='{}/shape_adapt_pool_{}'.format(self.name, i), parent=pii,
                                              kernel=self._shape_adaptation[i], stride=self._shape_adaptation[i]))
                projected_inputs[i] = self[-1]

        #in case of a reducing cell, we need to perform another max-pooling on all inputs
        if self._reducing:
            for i, pii in enumerate(projected_inputs):
                self.append(smo.MaxPool(name='{}/reduce_pool_{}'.format(self.name, i), parent=pii,
                                              kernel=(2,2), stride=(2,2)))
                projected_inputs[i] = self[-1]

        cell_modules=projected_inputs
        for i in range(self._n_modules):
            self.append(ZophBlock(name='{}/zoph_block_{}'.format(self.name, i), parents=cell_modules,
                                      candidates=self._candidates, channels=self._channels,
                                      join_parameters=self._join_parameters[i]))
            cell_modules.append(self[-1])

        #perform output concatenation
        self.append(smo.Merging(name=self.name+'/output_concat', parents=cell_modules, mode='concat'))

class ZophNetwork(smo.Graph):
    def __init__(self, name, parents, n_classes=10, stem_channels=64,
                 cells=[ZophCell]*3, cell_depth=[7]*3, cell_channels=[64, 64, 128],
                 reducing=[True]*3, join_parameters=[[None]*7]*3, candidates=ZOPH_CANDIDATES):
        self._n_classes = n_classes
        self._stem_channels = stem_channels
        self._cells = cells
        self._cell_depth = cell_depth
        self._cell_channels = cell_channels
        self._join_parameters = join_parameters
        self._reducing = reducing
        self._candidates = candidates

        smo.Graph.__init__(self, name, parents)

        #1. add the stem convolutions
        self.append(smo.Conv(name='{}/stem_conv_1'.format(self.name), parent=self.parent[0],
                             in_channels=self.parent[0].shape[1], out_channels=self._stem_channels,
                             kernel=(3,3), pad=(1,1)))
        self.append(smo.Conv(name='{}/stem_conv_2'.format(self.name), parent=self[-1],
                             in_channels=self._stem_channels, out_channels=self._stem_channels,
                             kernel=(3,3), pad=(1,1)))

        #2. add the cells using shared architecture parameters
        for i, celli in enumerate(zip(self._cells, self._cell_depth, self._cell_channels,
                                      self._join_parameters, self._reducing)):
            self.append(celli[0](name='{}/cell_{}'.format(self.name, i),
                                 parents = self[-2:],
                                 candidates=self._candidates,
                                 n_modules=celli[1],
                                 channels=celli[2],
                                 join_parameters=celli[3],
                                 reducing=celli[4]))

        #3. add output convolutions and global average pooling layers
        self.append(smo.Conv(name='{}/output_conv_1'.format(self.name), parent=self[-1],
                             in_channels=self[-1].shape[1], out_channels=self._n_classes,
                             kernel=(1,1)))
        self.append(smo.GlobalAvgPool(name='{}/global_average_pool'.format(self.name), parent=self[-1]))


    def get_arch_modules(self):
        ans = []
        for name, module in self.get_modules():
            if isinstance(module, smo.Join):
                ans.append(module)
        return ans


if __name__ == '__main__':
    import nnabla

    ctx = nnabla.ext_utils.get_extension_context('cudnn', device_id='0')
    nn.set_default_context(ctx)

    input = smo.Input(name='input', value=nn.Variable((10,20,32,16)))
    input2 = smo.Input(name='input', value=nn.Variable((10,20,32,32)))
    sep_conv3x3 = SepConv3x3(name='conv1', parent=input, channels=30)
    sep_conv5x5 = SepConv5x5(name='conv2', parent=input, channels=30)
    dil_sep_conv3x3 = DilSepConv3x3(name='conv3', parent=input, channels=30)
    dil_sep_conv5x5 = DilSepConv5x5(name='conv4', parent=input, channels=30)
    max_pool3x3 = MaxPool3x3(name='pool1', parent=input)
    avg_pool3x3 = AveragePool3x3(name='pool2', parent=input)
    zoph_block = ZophBlock(name='block1', parents=[input, input], channels=20, candidates=ZOPH_CANDIDATES)
    zoph_cell = ZophCell(name='cell1', parents=[input, input2], candidates=ZOPH_CANDIDATES, channels=32)
    zoph_network = ZophNetwork(name='network1', parents=[input2])

    #---------------------test graph setup--------------------------
    out_1 = sep_conv3x3()
    out_2 = sep_conv5x5()
    out_3 = dil_sep_conv3x3()
    out_4 = dil_sep_conv5x5()
    out_5 = max_pool3x3()
    out_6 = avg_pool3x3()
    out_7 = zoph_block()
    out_8 = zoph_cell()
    out_9 = zoph_network()

    #forward twice, just to get rid of the setup time when profiling
    for i in range(2):
        print("Warmup fw pass {}".format(i))
        out_9.forward()

    #-------------------do the profiling results sum up to the correct value------------------
    from nnabla_nas.module.static import NNablaProfiler
    from nnabla.utils.profiler import GraphProfiler
    #-----profiling the whole graph at once---------
    total_latency = []
    summed_latency = []
    module_latencies = {ci: {} for ci in ZOPH_CANDIDATES}

    for i in range(10):
        prof = GraphProfiler(out_9, device_id=0, ext_name='cudnn', n_run=10)
        prof.run()
        total_latency.append(float(prof.result['forward_all']))

        profiler    = NNablaProfiler()
        latencies   = zoph_network.profile(profiler, n_run=10)
        summed_lat  = 0.0
        for li in latencies:
            summed_lat += latencies[li]

        summed_latency.append(summed_lat)

        #-----profiling each module individually--------
        mods = zoph_network.get_modules()
        for _,mi in mods:
            if isinstance(mi, smo.Module):
                for mli in module_latencies:
                    if isinstance(mi, mli):
                        try:
                            module_latencies[mli][mi.shape].append(mi._prof)
                        except:
                            module_latencies[mli][mi.shape] = [mi._prof]
                mi._prof = None #reset the profile

        print(i)

    print('latency when profiling the whole graph at once is {}/{}'.format(np.mean(np.array(total_latency)), np.std(np.array(total_latency))))
    print('latency when profiling module by module is {}/{}'.format(np.mean(np.array(summed_latency)), np.std(np.array(summed_latency))))

    for mi in module_latencies:
        for si in module_latencies[mi]:
            print("Latency of module {} with shape {} is {}/{}".format(mi, si, np.mean(np.array(module_latencies[mi][si])), np.std(np.array(module_latencies[mi][si]))))

    import pdb; pdb.set_trace()
    exp_lat     = zoph_network.get_exp_latency(profiler)
    import pdb; pdb.set_trace()

