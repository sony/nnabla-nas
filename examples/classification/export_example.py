import sys
import os
import nnabla as nn
import glob

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


def export_all(exp_nr):
    
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

        OUTPUT_DIR = './logs/zoph/one_net/'
        
        # Sample one ZOPH network from the search space
        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)
        zn = zoph.SearchNet()
        output = zn(input)
        
        #zn_unique_active_modules = get_active_and_profiled_modules(zn)

        # GRAPH PDF
        zn.save_graph      (OUTPUT_DIR + 'zn')

        # WHOLE NET incl. latency
        zn.save_net_nnp    (OUTPUT_DIR + 'zn', input, output, save_latency=True)

        # MODULES incl. latency
        zn.save_modules_nnp(OUTPUT_DIR + 'zn', active_only=True, save_latency=True)
        
        # CONVERT TO ONNX
        zn.convert_npp_to_onnx(OUTPUT_DIR)

        # VERBOSITY - INFO OF NETWORK CONTENT
        #with open(OUTPUT_DIR + 'zn.txt', 'w') as f:
        #    print_me(zn, f)

    #  2 **************************
    if exp_nr == 2:
        from nnabla_nas.contrib import zoph

        OUTPUT_DIR = './logs/zoph/manynets/'
        
        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)
        N = 10  # number of random networks to sample

        # Sample N zoph networks from the search space
        for i in range(0,N):
            zn = zoph.SearchNet()
            output = zn(input)
            zn.save_graph      (OUTPUT_DIR + 'zn' + str(i))
            zn.save_net_nnp    (OUTPUT_DIR + 'zn' + str(i), input, output, save_latency=True)
            zn.save_modules_nnp(OUTPUT_DIR + 'zn' + str(i), active_only=True, save_latency=True)
        
        zn.convert_npp_to_onnx(OUTPUT_DIR)

    #  20 **************************
    if exp_nr == 20:
        from nnabla_nas.contrib import zoph
        import time

        OUTPUT_DIR = './logs/zoph/test_8/'
        
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
            print('****************** RUN ********************')
            zn.save_net_nnp    (OUTPUT_DIR + 'zn' + str(i), input, output, save_latency=True)
            zn.save_modules_nnp(OUTPUT_DIR + 'zn' + str(i), active_only=True, save_latency=True)
        zn.save_graph (OUTPUT_DIR)
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

        OUTPUT_DIR = './logs/rdn/one_net/'

        # Sample one random wired network from the search space
        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)
        rw = random_wired.TrainNet()
        output = rw(input)
        
        rw.save_graph      (OUTPUT_DIR + 'rw')
        rw.save_net_nnp    (OUTPUT_DIR + 'rw', input, output)
        rw.save_modules_nnp(OUTPUT_DIR + 'rw', active_only=True)
        rw.convert_npp_to_onnx(OUTPUT_DIR)

    #  40 **************************
    if exp_nr == 40:
        from nnabla_nas.contrib import random_wired
        import time

        OUTPUT_DIR = './logs/rdn/test_2/'
        
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

        N = 10  # Measure latency on same rdn network N times
        for i in range(0,N):
            print('****************** RUN ********************')
            rw.save_net_nnp    (OUTPUT_DIR + 'rw' + str(i), input, output, save_latency=True)
            rw.save_modules_nnp(OUTPUT_DIR + 'rw' + str(i), active_only=True)
        rw.save_graph      (OUTPUT_DIR + 'rw' + str(i))
        rw.convert_npp_to_onnx(OUTPUT_DIR)

    #  5 **************************
    if exp_nr == 5:
        from nnabla_nas.contrib.classification.mobilenet import SearchNet

        OUTPUT_DIR = './logs/mobilenet/app46/'

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

        OUTPUT_DIR = './logs/mobilenet/test_2/'

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


        N = 10  # Measure latency on same network N times
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

        #INPUT_DIR = './logs/zoph/0_app74busy_many_nets/'
        #INPUT_DIR = './logs/zoph/1_app46free_many_nets/'
        #INPUT_DIR = './logs/zoph/2_app46free_many_nets/'
        #INPUT_DIR = './logs/zoph/3_app79free_many_nets/'
        #INPUT_DIR = './logs/zoph/app79cpu/'
        #INPUT_DIR = './logs/zoph/snpe_machine_test/'
        INPUT_DIR = './logs/zoph/test_8/'

        existing_networks = glob.glob(INPUT_DIR + '/*' + os.path.sep)
        all_nets_latencies = dict.fromkeys([])
        all_nets = dict.fromkeys([])
        net_idx = 0
        for network in existing_networks:
            all_blocks = glob.glob(network + '**/*.onnx', recursive=True)
            blocks = dict.fromkeys([])
            block_idx = 0
            this_net_accumulated_latency = 0
            for block in all_blocks:
                print('.... READING .... -->  ' + block)

                # Interesting FIELDS in params.graph: 'input', 'name', 'node', 'output'
                params = onnx.load(block)

                # Reading latency for each of the blocks of layers
                block_lat = block[:-5] + '.lat'
                with open(block_lat, 'r') as f:
                    block_latency = float(f.read())

                this_net_accumulated_latency += block_latency

                this_block = dict.fromkeys([])
                this_block['latency'] = block_latency
                this_block['name']    = params.graph.name
                this_block['input']   = params.graph.input
                this_block['output']  = params.graph.output
                this_block['nodes']   = params.graph.node
                blocks[block_idx] = this_block
                block_idx += 1

            net_file  = network[:-1] + '.onnx'
            print('xxxx READING xxxx -->  ' + net_file)
            params = onnx.load(net_file)

            net_lat_file = network[:-1] + '.lat'
            with open(net_lat_file, 'r') as f:
                this_net_latency = float(f.read())

            this_net = dict.fromkeys([])
            this_net['latency'] = this_net_latency
            this_net['accum_latency'] = this_net_accumulated_latency
            this_net['name']    = params.graph.name
            this_net['input']   = params.graph.input
            this_net['output']  = params.graph.output
            this_net['nodes']   = params.graph.node

            all_nets_latencies[net_idx] = [this_net_latency, this_net_accumulated_latency]
            all_nets          [net_idx] = this_net

            net_idx += 1

        # Compare accumulated latency to net latencies, do a plot:
        import pdb; pdb.set_trace()

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
    # import pdb; pdb.set_trace()
    if len(sys.argv) > 1:
        export_all(int(sys.argv[1]))
    else:
        print('Usage: python export_example.py <NUM>')
        print('Possible values for NUM:')
        print('# 0 : sandbox -  creation / exporting tests')
        print('# 1 : create 1 instance of ZOPH network,                      save it and its modules,     convert to ONNX')
        print('# 2 : Sample a set of N ZOPH networks,         export all of them (whole net and modules), convert to ONNX')
        print('# 20 : Sample one ZOPH network, calculate latency of this network N times')
        print('# 21 : Sample several static convolutions (predefined), calc latency on each of them many times')
        print('# 22 : Sample several random-sized static convolutions. Calc latency on each of them many times')                
        print('# 3 : create 1 instance of random wired search space network, save it and its modules,     convert to ONNX')
        print('# 4 : Sample a set of N Random wired networks, export them (net and modules), convert to ONNX')
        print('# 40 : Sample one Random wired network, calculate latency of this network N times, convert to ONNX')
        print('# 5 : WIP: the export for dynamic modules using mobilenet')
        print('# 50 : Sample one mobilenet network, calculate latency N times')        
        print('# 6 : WIP: load ONNXs, load latencies, put everything on dictionary')
        print('# 7 : WIP: load nnp files')
    pass

    