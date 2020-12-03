import sys
import os
import nnabla as nn
import glob
from nnabla.ext_utils import get_extension_context

# Global params for the latency estimation
outlier = 0.05
max_measure_execution_time = 500
time_scale = "m"
n_warmup = 10
n_runs = 100

estim_accum=None
estim_real =None


def print_me(zn,f):
    print(zn.summary(), file=f)
    print('\n***************************\n', file=f)
    print(len(zn.get_arch_modules()), file=f)
    print(zn.get_arch_modules(), file=f)
    print('\n***************************\n', file=f)
    print(len(zn.get_net_modules(active_only=False)), file=f)
    print(zn.get_net_modules(active_only=False), file=f)
    print('\n***************************\n', file=f)
    print(len(zn.get_net_modules(active_only=True)), file=f)
    print(zn.get_net_modules(active_only=True), file=f)


def get_active_and_profiled_modules(zn):
    print('\n**START PRINT ******************************')
    list = []
    for mi in zn.get_net_modules(active_only=True):
        if type(mi) in zn.modules_to_profile:
            if type(mi) not in list:
                print(type(mi))        
                list.append(type(mi)) 
    print('**END PRINT ******************************\n')
    return list


def estim_fwd(output, n_run=100):
    import time
    # warm up
    output.forward()
    result = 0.0
    #start_0 = time.time()
    for j in range(n_run):
        start = time.time()
        output.forward()
        stop = time.time()
        result += stop - start

    return result / n_run

def init_calc_latency(output, ext_name='cpu', device_id=0):
    from nnabla_nas.utils.estimator.latency import LatencyGraphEstimator, Profiler
    estim_accum = LatencyGraphEstimator(
        device_id=device_id, ext_name=ext_name,
        outlier=outlier,
        time_scale=time_scale,
        n_warmup=n_warmup,
        max_measure_execution_time=max_measure_execution_time,
        n_run=n_runs
    )

    estim_real = Profiler(output,
        device_id=device_id, ext_name=ext_name,
        outlier=outlier,
        time_scale=time_scale,
        n_warmup=n_warmup,
        max_measure_execution_time=max_measure_execution_time,
        n_run=n_runs
    )
    return estim_real, estim_accum



def export_all(exp_nr, calc_latency=False, ext_name='cpu', device_id=0, onnx=False):
    
    #  0 **************************
    if exp_nr == 0:
        from nnabla_nas.contrib import zoph

        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)
        
        zn1 = zoph.SearchNet()

        zn1a_unique_active_modules = get_active_and_profiled_modules(zn1)
        zn1.save_graph('./logs/sandbox/zn1a')
        zn1.save_modules_nnp('./logs/sandbox/zn1a', active_only=True)
        with open('./logs/sandbox/zn1a.txt', 'w') as f:
            print_me(zn1, f)

        out1b = zn1(input)
        zn1b_unique_active_modules = get_active_and_profiled_modules(zn1)
        zn1.save_graph('./logs/sandbox/zn1b')
        zn1.save_modules_nnp('./logs/sandbox/zn1b', active_only=True)
        with open('./logs/sandbox/zn1b.txt', 'w') as f:
            print_me(zn1, f)
    
        out1c = zn1(input)
        zn1c_unique_active_modules = get_active_and_profiled_modules(zn1)
        zn1.save_graph('./logs/sandbox/zn1c')
        zn1.save_modules_nnp('./logs/sandbox/zn1c', active_only=True)
        with open('./logs/sandbox/zn1c.txt', 'w') as f:
            print_me(zn1, f)

    #  1 **************************
    if exp_nr == 1:
        from nnabla_nas.contrib import zoph
        ctx = get_extension_context(ext_name=ext_name, device_id=device_id)
        nn.set_default_context(ctx)

        OUTPUT_DIR = './logs/zoph/one_net/'
        
        # Sample one ZOPH network from the search space
        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)
        zn = zoph.SearchNet()
        output = zn(input)

        if calc_latency:
            estim_real, estim_accum = init_calc_latency(output, ext_name=ext_name, device_id=device_id)

        #zn_unique_active_modules = get_active_and_profiled_modules(zn)

        # SAVE GRAPH in PDF
        zn.save_graph(OUTPUT_DIR + 'zn')

        # SAVE WHOLE NET
        zn.save_net_nnp(OUTPUT_DIR + 'zn', input, output, 
            calc_latency=calc_latency, func_real_latency=estim_real, func_accum_latency=estim_accum)

        # SAVE ALL MODULES 
        zn.save_modules_nnp(OUTPUT_DIR + 'zn', active_only=True, 
            calc_latency=calc_latency, func_latency=estim_accum)

        # CONVERT ALL TO ONNX
        if onnx:
            zn.convert_npp_to_onnx(OUTPUT_DIR)

        # VERBOSITY - INFO OF NETWORK CONTENT
        #with open(OUTPUT_DIR + 'zn.txt', 'w') as f:
        #    print_me(zn, f)
        #import pdb; pdb.set_trace()

    #  2 **************************
    if exp_nr == 2:
        from nnabla_nas.contrib import zoph
        ctx = get_extension_context(ext_name=ext_name, device_id=device_id)
        nn.set_default_context(ctx)

        OUTPUT_DIR = './logs/zoph/many_different_nets/'

        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)

        N = 10  # number of random networks to sample

        # Sample N zoph networks from the search space
        for i in range(0,N):
            zn = zoph.SearchNet()
            output = zn(input)
            if calc_latency:
                estim_real, estim_accum = init_calc_latency(output, ext_name=ext_name, device_id=device_id)

            zn.save_graph      (OUTPUT_DIR + 'zn' + str(i))
            zn.save_net_nnp    (OUTPUT_DIR + 'zn' + str(i), input, output, 
                calc_latency=calc_latency, func_real_latency=estim_real, func_accum_latency=estim_accum)
            zn.save_modules_nnp(OUTPUT_DIR + 'zn' + str(i), active_only=True, 
                calc_latency=calc_latency, func_latency=estim_accum)
        
        if onnx:
            zn.convert_npp_to_onnx(OUTPUT_DIR)

    #  20 **************************
    if exp_nr == 20:
        from nnabla_nas.contrib import zoph
        import time
        ctx = get_extension_context(ext_name=ext_name, device_id=device_id)
        nn.set_default_context(ctx)

        OUTPUT_DIR = './logs/zoph/same_net_repeated/'
        
        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)
        zn = zoph.SearchNet()
        output = zn(input)


        N = 5 # Measure add-hoc latency of zoph network
        for i in range(0,N):
            n_run = 100
            # warm up
            output.forward()

            result = 0.0
            start_0 = time.time()
            for i in range(n_run):
                start = time.time()
                output.forward()
                stop = time.time()
                result += stop - start

            mean_time = result / n_run
            print(mean_time*1000)
        
        N = 10  # Measure latency on same zoph network N times using LatencyEstimator or LatencyGraphEstimator
        for i in range(0,N):
            if calc_latency:
                estim_real, estim_accum = init_calc_latency(output, ext_name=ext_name, device_id=device_id)
            zn.save_net_nnp    (OUTPUT_DIR + 'zn' + str(i), input, output, 
                calc_latency=calc_latency, func_real_latency=estim_real, func_accum_latency=estim_accum)
            zn.save_modules_nnp(OUTPUT_DIR + 'zn' + str(i), active_only=True, 
                calc_latency=calc_latency, func_latency=estim_accum)
        zn.save_graph (OUTPUT_DIR)
        if onnx:
            zn.convert_npp_to_onnx(OUTPUT_DIR)
    
    #  21 **************************
    if exp_nr == 21:    
        from nnabla_nas.module import static as smo

        input1 = nn.Variable((1, 256, 32, 32))
        input2 = nn.Variable((1, 384, 32, 32))
        input3 = nn.Variable((1, 128, 32, 32))
        input4 = nn.Variable((1, 768, 32, 32))
        input5 = nn.Variable((1, 1280, 32, 32))
        input6 = nn.Variable((1, 2048, 32, 32))
        input7 = nn.Variable((1, 512, 32, 32))
        input8 = nn.Variable((1, 192, 32, 32))
        input9 = nn.Variable((1, 224, 32, 32))
        
        static_input1 = smo.Input(value=input1)
        static_input2 = smo.Input(value=input2)
        static_input3 = smo.Input(value=input3)
        static_input4 = smo.Input(value=input4)
        static_input5 = smo.Input(value=input5)
        static_input6 = smo.Input(value=input6)
        static_input7 = smo.Input(value=input7)
        static_input8 = smo.Input(value=input8)
        static_input9 = smo.Input(value=input9)

        myconv1 = smo.Conv(parents=[static_input1], in_channels=256, out_channels=128, kernel=(1,1), pad=None, group=1)
        myconv2 = smo.Conv(parents=[static_input2], in_channels=384, out_channels=128, kernel=(1,1), pad=None, group=1)
        myconv3 = smo.Conv(parents=[static_input3], in_channels=128, out_channels=256, kernel=(1,1), pad=None, group=1)
        myconv4 = smo.Conv(parents=[static_input4], in_channels=768, out_channels=256, kernel=(1,1))
        myconv5 = smo.Conv(parents=[static_input5], in_channels=1280, out_channels=256, kernel=(1,1), pad=None, group=1)
        myconv6 = smo.Conv(parents=[static_input6], in_channels=2048, out_channels=256, kernel=(1,1), pad=None, group=1)
        myconv7 = smo.Conv(parents=[static_input7], in_channels=512, out_channels=512, kernel=(3,3), pad=(1,1), group=1)
        myconv8 = smo.Conv(parents=[static_input8], in_channels=192, out_channels=512, kernel=(7,7), pad=(3,3), group=1)
        myconv9 = smo.Conv(parents=[static_input9], in_channels=224, out_channels=128, kernel=(5,5), pad=(2,2), group=1)

        output1 = myconv1()
        output2 = myconv2()
        output3 = myconv3()
        output4 = myconv4()
        output5 = myconv5()
        output6 = myconv6()
        output7 = myconv7()
        output8 = myconv8()
        output9 = myconv9()

        N = 10 
        for i in range(0,N):
            mean_time = estim_fwd(output1)
            print("1, ", mean_time)
            mean_time = estim_fwd(output2)
            print("2, ", mean_time)
            mean_time = estim_fwd(output3)
            print("3, ", mean_time)
            mean_time = estim_fwd(output4)
            print("4, ", mean_time)
            mean_time = estim_fwd(output5)
            print("5, ", mean_time)
            mean_time = estim_fwd(output6)
            print("6, ", mean_time)

        N = 100
        for i in range(0,N):
            print('****************** RUN ********************')
            from nnabla_nas.utils.estimator import LatencyEstimator
            estimation = LatencyEstimator(n_run = 100, ext_name='cpu')
            latency = estimation.get_estimation(myconv1)
            latency = estimation.get_estimation(myconv2)
            latency = estimation.get_estimation(myconv3)
            latency = estimation.get_estimation(myconv4)
            latency = estimation.get_estimation(myconv5)
            latency = estimation.get_estimation(myconv6)
            latency = estimation.get_estimation(myconv7)
            latency = estimation.get_estimation(myconv8)
            latency = estimation.get_estimation(myconv9)

            estimation = LatencyEstimator(n_run = 100, ext_name='cpu')
            latency = estimation.get_estimation(myconv9)
            latency = estimation.get_estimation(myconv8)
            latency = estimation.get_estimation(myconv7)
            latency = estimation.get_estimation(myconv6)
            latency = estimation.get_estimation(myconv5)
            latency = estimation.get_estimation(myconv4)
            latency = estimation.get_estimation(myconv3)
            latency = estimation.get_estimation(myconv2)
            latency = estimation.get_estimation(myconv1)

            estimation = LatencyEstimator(n_run = 100, ext_name='cpu')
            latency = estimation.get_estimation(myconv6)
            latency = estimation.get_estimation(myconv9)
            latency = estimation.get_estimation(myconv1)
            latency = estimation.get_estimation(myconv4)
            latency = estimation.get_estimation(myconv8)
            latency = estimation.get_estimation(myconv3)
            latency = estimation.get_estimation(myconv5)
            latency = estimation.get_estimation(myconv7)
            latency = estimation.get_estimation(myconv2)

    #  22 **************************
    if exp_nr == 22:
        from nnabla_nas.module import static as smo
        from nnabla_nas.utils.estimator import LatencyEstimator
        from numpy.random import permutation
        import numpy as np
        
        run_also_ours_at_the_end = True

        N_conv = 50 # number of different convolutions tried
        in_sizes  = np.random.randint(low=1, high=1000, size=N_conv)
        out_sizes  = np.random.randint(low=1, high=600, size=N_conv)
        kernel_sizes = np.random.randint(low=1, high=7, size=N_conv)
        feat_sizes = np.random.randint(low=16, high=48, size=N_conv)
        
        N = 100
        for j in range(N):
            estimation = LatencyEstimator(n_run = 100, ext_name='cpu')
            print('****************** RUN ********************')
            for i in permutation(N_conv):
                input = nn.Variable((1, in_sizes[i], feat_sizes[i], feat_sizes[i]))
                static_input = smo.Input(value=input)
                myconv = smo.Conv(parents=[static_input], in_channels=in_sizes[i], out_channels=out_sizes[i],
                                 kernel=(kernel_sizes[i],kernel_sizes[i]), pad=None, group=1)
                output = myconv()
                latency = estimation.get_estimation(myconv)

        if run_also_ours_at_the_end is True:
            print('*********** NOW IT IS OUR TURN ***********')
            for i in range(N_conv):
                input = nn.Variable((1, in_sizes[i], feat_sizes[i], feat_sizes[i]))
                static_input = smo.Input(value=input)
                myconv = smo.Conv(parents=[static_input], in_channels=in_sizes[i], out_channels=out_sizes[i],
                                 kernel=(kernel_sizes[i],kernel_sizes[i]), pad=None, group=1)
                output = myconv()
                mean_time = estim_fwd(output, n_run=100) * 1000 # in ms
                print('Our_Conv : 100 :', mean_time, ':', 
                     '[(1, '+str(in_sizes[i])+', '+str(feat_sizes[i])+', '+str(feat_sizes[i])+')]', 
                     ':', out_sizes[i], ':', kernel_sizes[i]
                     )

    #  3 **************************
    if exp_nr == 3:
        from nnabla_nas.contrib import random_wired
        ctx = get_extension_context(ext_name=ext_name, device_id=device_id)
        nn.set_default_context(ctx)

        OUTPUT_DIR = './logs/rdn/one_net/'

        # Sample one random wired network from the search space
        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)
        rw = random_wired.TrainNet()
        output = rw(input)

        if calc_latency:
            estim_real, estim_accum = init_calc_latency(output, ext_name=ext_name, device_id=device_id)

        rw.save_graph      (OUTPUT_DIR + 'rw')
        rw.save_net_nnp    (OUTPUT_DIR + 'rw', input, output,
            calc_latency=calc_latency, func_real_latency=estim_real, func_accum_latency=estim_accum)
        rw.save_modules_nnp(OUTPUT_DIR + 'rw', active_only=True,
            calc_latency=calc_latency, func_latency=estim_accum)
        
        if onnx:
            rw.convert_npp_to_onnx(OUTPUT_DIR)

    #  4 **************************    
    if exp_nr == 4:
        from nnabla_nas.contrib import random_wired
        ctx = get_extension_context(ext_name=ext_name, device_id=device_id)
        nn.set_default_context(ctx)

        OUTPUT_DIR = './logs/rdn/many_different_nets/'
        
        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)

        N = 10  # Measure latency on same rdn network N times
        for i in range(0,N):
            rw = random_wired.TrainNet()
            output = rw(input)
            if calc_latency:
                estim_real, estim_accum = init_calc_latency(output, ext_name=ext_name, device_id=device_id)
            
            rw.save_graph      (OUTPUT_DIR + 'rw' + str(i))
            rw.save_net_nnp    (OUTPUT_DIR + 'rw' + str(i), input, output,
                calc_latency=calc_latency, func_real_latency=estim_real, func_accum_latency=estim_accum)
            rw.save_modules_nnp(OUTPUT_DIR + 'rw' + str(i), active_only=True, 
                calc_latency=calc_latency, func_latency=estim_accum)
        
        if onnx:
            rw.convert_npp_to_onnx(OUTPUT_DIR)

    #  40 **************************
    if exp_nr == 40:
        from nnabla_nas.contrib import random_wired
        import time

        OUTPUT_DIR = './logs/rdn/same_net_many_times/'
        ctx = get_extension_context(ext_name=ext_name, device_id=device_id)
        nn.set_default_context(ctx)

        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)
        rw = random_wired.TrainNet()
        output = rw(input)

        N = 5 
        for i in range(0,N):
            n_run = 10
            # warm up
            output.forward()

            result = 0.0
            start_0 = time.time()
            for i in range(n_run):
                start = time.time()
                output.forward()
                stop = time.time()
                result += stop - start
            mean_time = result / n_run
            print(mean_time*1000)

        N = 5  # Measure latency on same rdn network N times
        for i in range(0,N):
            if calc_latency:
                estim_real, estim_accum = init_calc_latency(output, ext_name=ext_name, device_id=device_id)
            rw.save_net_nnp    (OUTPUT_DIR + 'rw' + str(i), input, output,
                calc_latency=calc_latency, func_real_latency=estim_real, func_accum_latency=estim_accum)
            rw.save_modules_nnp(OUTPUT_DIR + 'rw' + str(i), active_only=True,
                calc_latency=calc_latency, func_latency=estim_accum)
        rw.save_graph      (OUTPUT_DIR + 'rw' + str(i))
        
        if onnx:
            rw.convert_npp_to_onnx(OUTPUT_DIR)

    #  5 **************************
    if exp_nr == 5:
        from nnabla_nas.contrib.classification.mobilenet import SearchNet

        OUTPUT_DIR = './logs/mobilenet/many_different_nets/'

        input = nn.Variable((1, 3, 224, 224))
        
        N = 10  # number of random networks to sample

        # Sample N networks from the search space
        for i in range(0,N):
            mobile_net = SearchNet(num_classes=1000)
            output = mobile_net(input)
            
            #mobile_net.save_net_nnp(OUTPUT_DIR + 'mn' + str(i), input, output, save_latency=True)
            
            # it seems the last affine layer cannot be converted to ONNX (!?), here export without it
            mobile_net.save_net_nnp(OUTPUT_DIR + 'mn' + str(i), input, output.parent.inputs[0], save_latency=True)
            
            #mobile_net.save_modules_nnp() ## NOT AVAILABLE YET
        
        mobile_net.convert_npp_to_onnx(OUTPUT_DIR)
    
    #  50 **************************
    if exp_nr == 50:
        from nnabla_nas.contrib.classification.mobilenet import SearchNet
        import time

        OUTPUT_DIR = './logs/mobilenet/same_net_many_times/'

        input = nn.Variable((1, 3, 224, 224))
        mobile_net = SearchNet(num_classes=1000)
        output = mobile_net(input)
        
        N = 5 
        for i in range(0,N):
            n_run = 100
            # warm up
            output.forward()

            result = 0.0
            start_0 = time.time()
            for i in range(n_run):
                start = time.time()
                output.forward()
                stop = time.time()
                result += stop - start

            mean_time = result / n_run
            print(mean_time*1000)


        N = 5  # Measure latency on same network N times
        for i in range(0,N):
            print('****************** RUN ********************')
            mobile_net.save_net_nnp(OUTPUT_DIR + 'mn' + str(i), input, output, save_latency=True)
        
        #mobile_net.save_modules_nnp()
        # it seems the last affine layer cannot be converted to ONNX (!?), here export without it
        #mobile_net.save_net_nnp(OUTPUT_DIR + 'mn' + str(i), input, output.parent.inputs[0], save_latency=True)
        #mobile_net.convert_npp_to_onnx(OUTPUT_DIR)

    #  6 **************************        
    if exp_nr == 6:
        import onnx
        load_onnx = False

        if len(sys.argv) > 2:
            INPUT_DIR = sys.argv[2]
        else:  
            #INPUT_DIR = './logs/zoph/0_app74busy_many_nets/'
            #INPUT_DIR = './logs/zoph/1_app46free_many_nets/'
            #INPUT_DIR = './logs/zoph/2_app46free_many_nets/'
            #INPUT_DIR = './logs/zoph/3_app79free_many_nets/'
            #INPUT_DIR = './logs/zoph/app79cpu/'
            #INPUT_DIR = './logs/zoph/snpe_machine_test/'
            INPUT_DIR = './logs/zoph/many_different_nets/'

        if len(sys.argv) > 3:
            load_onnx = True


        existing_networks = glob.glob(INPUT_DIR + '/*' + os.path.sep)
        all_nets_latencies = dict.fromkeys([])
        all_nets = dict.fromkeys([])
        net_idx = 0
        for network in existing_networks:
            all_blocks = glob.glob(network + '**/*.acclat', recursive=True)
            blocks = dict.fromkeys([])
            block_idx = 0

            this_net_accumulated_latency = 0.0
            this_net_accumulated_latency_of_convs = 0.0
            this_net_accumulated_latency_of_relus = 0.0
            this_net_accumulated_latency_of_bns = 0.0
            this_net_accumulated_latency_of_merges = 0.0
            this_net_accumulated_latency_of_pools = 0.0
            this_net_accumulated_latency_of_reshapes = 0.0

            for block_lat in all_blocks:
                block = block_lat[:-7] + '.onnx'
                print('.... READING .... -->  ' + block)
    

                # Reading latency for each of the blocks of layers
                with open(block_lat, 'r') as f:
                    block_latency = float(f.read())

                this_net_accumulated_latency += block_latency

                # Layer-type-wise latencies tested for Zoph and for Random Wired networks
                layer_name = block.split('/')[-1].split('.')[0]
                if layer_name.find('bn') != -1:
                    this_net_accumulated_latency_of_bns += block_latency
                elif layer_name.find('relu') != -1:
                    this_net_accumulated_latency_of_relus += block_latency
                elif layer_name.find('conv') != -1:
                    this_net_accumulated_latency_of_convs += block_latency
                elif layer_name.find('merg') != -1:
                    this_net_accumulated_latency_of_merges += block_latency
                elif layer_name.find('pool') != -1:
                    this_net_accumulated_latency_of_pools += block_latency
                elif layer_name.find('con') != -1: # from concat
                    this_net_accumulated_latency_of_merges += block_latency
                elif layer_name.find('reshape') != -1:
                    this_net_accumulated_latency_of_reshapes += block_latency
                pass

                this_block = dict.fromkeys([])
                this_block['latency'] = block_latency

                if load_onnx:
                    # Interesting FIELDS in params.graph: 'input', 'name', 'node', 'output'
                    params = onnx.load(block)
                    this_block['name']    = params.graph.name
                    this_block['input']   = params.graph.input
                    this_block['output']  = params.graph.output
                    this_block['nodes']   = params.graph.node

                blocks[block_idx] = this_block
                block_idx += 1


            net_realat_file = network[:-1] + '.realat'
            with open(net_realat_file, 'r') as f:
                this_net_real_latency = float(f.read())

            net_acclat_file = network[:-1] + '.acclat'
            with open(net_acclat_file, 'r') as f:
                this_net_acc_latency = float(f.read())
            
            this_net = dict.fromkeys([])
            this_net['real_latency'] = this_net_real_latency
            this_net['accum_latency_graph'] = this_net_acc_latency
            this_net['accum_latency_module'] = this_net_accumulated_latency
            
            if load_onnx:
                net_file  = network[:-1] + '.onnx'
                print('xxxx READING xxxx -->  ' + net_file)
                params = onnx.load(net_file)
                this_net['name']    = params.graph.name
                this_net['input']   = params.graph.input
                this_net['output']  = params.graph.output
                this_net['nodes']   = params.graph.node

            all_nets_latencies[net_idx] = [this_net_real_latency, this_net_acc_latency, this_net_accumulated_latency,
                                        this_net_accumulated_latency_of_convs,    this_net_accumulated_latency_of_bns, 
                                        this_net_accumulated_latency_of_relus,  this_net_accumulated_latency_of_pools, 
                                        this_net_accumulated_latency_of_merges, this_net_accumulated_latency_of_reshapes,
                                        ]
            all_nets          [net_idx] = this_net

            net_idx += 1

        # Compare accumulated latency to net latencies, do a plot:
        print('LATENCY Results from ' + INPUT_DIR)
        print('NETWORK REAL, ACCUM by graph analysis, ACCUM by module analysis,', 
               ' of CONVs,  of BNs,  of RELUs, of POOLs, of MERGEs/CONCATs, of RESHAPEs',
                )
        for i in range(len(all_nets_latencies)):
            #print(all_nets_latencies[i])
            print(['%7.4f' % val for val in all_nets_latencies[i]])

        #import pdb; pdb.set_trace()

    #  7 **************************        
    if exp_nr == 7:
        from nnabla.utils.nnp_graph import NnpLoader as load
        for filename in glob.glob('./logs/zoph/one_net/zn/**/*.nnp', recursive=True):
            if 'SepConv' in filename:
                import pdb; pdb.set_trace()

            print(filename)
            nnp = load(filename)
            net_name = nnp.get_network_names()[0]
            net = nnp.get_network(net_name)
        
            params = nnp.network_dict[list(nnp.network_dict.keys())[0]]

            print(net.inputs)
            print(net.outputs)
            for fx in params.function:
                #print(fx.type)
                print(fx)
            

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            if sys.argv[2] == '0' or sys.argv[2] == 'LO' or sys.argv[2] == 'OL':
                onnx=True
            else:
                onnx=False
            pass
            if sys.argv[2] == 'L' or sys.argv[2] == 'LO' or sys.argv[2] == 'OL':
                calc_latency=True
            else:
                calc_latency=False
            pass
            if sys.argv[2] == 'L' or sys.argv[2] == 'LO' or sys.argv[2] == 'OL' or sys.argv[2] == 'O':
                if len(sys.argv) > 3:
                    if len(sys.argv) > 4:
                        export_all(int(sys.argv[1]), calc_latency=calc_latency, ext_name=sys.argv[3], device_id=int(sys.argv[4]), onnx=onnx)    
                    else:
                        export_all(int(sys.argv[1]), calc_latency=calc_latency, ext_name=sys.argv[3], onnx=onnx)    
                    pass
                else:
                    export_all(int(sys.argv[1]), calc_latency=calc_latency, onnx=onnx)
                pass
            else:
                if len(sys.argv) > 3:
                    export_all(int(sys.argv[1]), calc_latency=calc_latency, ext_name=sys.argv[2], device_id=int(sys.argv[3]), onnx=onnx)    
                else:
                    export_all(int(sys.argv[1]), calc_latency=calc_latency, ext_name=sys.argv[2], onnx=onnx)    
                pass
            pass
        else:
            export_all(int(sys.argv[1]), calc_latency=False)

    else:
        print('Usage: python export_example.py <id> [L|LO|O|<path>] [<ext_name> [<device-id>]]')
        print('If L is used, the estimation for latency will be calculated')
        print('If O is used, the exporting to ONNX will be done')
        print('If LO or OL is used, the estimation and the exporting to ONNX will be done')
        print('Possible values for <ext_name>: cpu, cuda, cudnn. Default = cpu')
        print('Possible values for <device_id>: 0..7 . Default = 0')
        print('Possible values for <id>:')
        print('# 0  : sandbox -  creation / exporting tests')
        print('# 1  : (one net) create 1 instance of ZOPH network,   save it and its modules,  calc latency, convert to ONNX')
        print('# 2  : (many different nets) Sample a set of N ZOPH networks, calculate latency, export all of them (whole net and modules), convert to ONNX')
        print('# 20 : (one net many times) Sample one ZOPH network, calculate latency of this network N times')
        print('# 21 : Sample several static convolutions (predefined), calc latency on each of them many times')
        print('# 22 : Sample several random-sized static convolutions. Calc latency on each of them many times')                
        print('# 3  : (one net) create 1 instance of random wired search space network, save it and its modules,     convert to ONNX')
        print('# 4  : (many different nets) Sample a set of N Random wired networks, export them (net and modules), convert to ONNX')
        print('# 40 : (one net many times) Sample one Random wired network, calculate latency of this network N times, convert to ONNX')
        print('# 5  : (many different nets) WIP: the export for dynamic modules using mobilenet')
        print('# 50 : (one net many times)  WIP: Sample one mobilenet network, calculate latency N times')        
        print('# 6 <path>  (do not use ext_name or device_id): (WIP) from the given <path>, load ONNXs, load latencies, put everything on dictionary, display it')
        print('# 7 : WIP: load nnp files')
    pass

    