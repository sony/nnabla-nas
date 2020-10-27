import sys
import os
import nnabla as nn
import glob
from nnabla_nas.utils.estimator import LatencyEstimator



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

def export_all(runme):
    
    #  0 **************************
    if runme is '0':
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
    if runme is '1':
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
    if runme is '2':
        from nnabla_nas.contrib import zoph

        OUTPUT_DIR = './logs/zoph/snpe_machine_test/'
        #OUTPUT_DIR = './logs/zoph/4_same_net_many_times/'
        
        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)
        N = 20  # number of random networks to sample
        zn = zoph.SearchNet()
        output = zn(input)

        # Sample N zoph networks from the search space
        for i in range(0,N):
            zn.save_graph      (OUTPUT_DIR + 'zn' + str(i))
            zn.save_net_nnp    (OUTPUT_DIR + 'zn' + str(i), input, output, save_latency=True)
            zn.save_modules_nnp(OUTPUT_DIR + 'zn' + str(i), active_only=True, save_latency=True)
        
        zn.convert_npp_to_onnx(OUTPUT_DIR)

    #  3 **************************
    if runme is '3':
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


    #  4 **************************
    if runme is '4':
        from nnabla_nas.contrib import random_wired

        OUTPUT_DIR = './logs/rdn/many_nets/'
        
        shape = (1, 3, 32, 32)
        input = nn.Variable(shape)
        N = 5  # number of random networks to sample

        # Sample N random wired networks from the search space
        for i in range(0,N):
            rw = random_wired.TrainNet()
            output = rw(input)
            rw.save_graph      (OUTPUT_DIR + 'rw' + str(i))
            rw.save_net_nnp    (OUTPUT_DIR + 'rw' + str(i), input, output)
            rw.save_modules_nnp(OUTPUT_DIR + 'rw' + str(i), active_only=True)
        rw.convert_npp_to_onnx(OUTPUT_DIR)

    #  5 **************************
    if runme is '5':
        from nnabla_nas.contrib.classification.mobilenet import SearchNet

        OUTPUT_DIR = './logs/output_runme_5/'
        
        mobile_net = SearchNet(num_classes=1000)
        input = nn.Variable((1, 3, 224, 224))
        output = mobile_net(input)
        
        #mobile_net.save_modules_nnp(OUTPUT_DIR + 'mn', True)

        # mobile_net.save_net_nnp(OUTPUT_DIR + 'mn', input, output)
        # it seems the last affine layer cannot be converted to ONNX (!?),
        # thus export without it
        mobile_net.save_net_nnp(OUTPUT_DIR + 'mn', input, output.parent.inputs[0])
        
        mobile_net.convert_npp_to_onnx(OUTPUT_DIR)
    
    #  6 **************************        
    if runme is '6':
        import onnx

        #INPUT_DIR = './logs/zoph/0_app74busy_many_nets/'
        #INPUT_DIR = './logs/zoph/1_app46free_many_nets/'
        #INPUT_DIR = './logs/zoph/2_app46free_many_nets/'
        #INPUT_DIR = './logs/zoph/3_app79free_many_nets/'
        #INPUT_DIR = './logs/zoph/2_same_net_many_times/'
        INPUT_DIR = './logs/zoph/snpe_machine_test/'
        
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
    if runme is '7':
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
        export_all(sys.argv[1])
    else:
        print('Usage: python export_example.py <NUM>')
        print('Possible values for NUM:')
        print('# 0 : sandbox -  creation / exporting tests')
        print('# 1 : create 1 instance of ZOPH network,                      save it and its modules,     convert to ONNX')
        print('# 2 : Sample a set of N ZOPH networks,         export all of them (whole net and modules), convert to ONNX')
        print('# 3 : create 1 instance of random wired search space network, save it and its modules,     convert to ONNX')
        print('# 4 : Sample a set of N RANDOM WIRED networks, export all of them (whole net and modules), convert to ONNX')
        print('# 5 : WIP: the export for dynamic modules')
        print('# 6 : WIP: load ONNXs, load latencies, put everything on dictionary')
        print('# 7 : WIP: load nnp files')
    pass

    