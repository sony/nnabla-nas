
import os
import nnabla as nn

from nnabla_nas.contrib import zoph

from nnabla.utils.nnp_graph import NnpLoader as load

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

def zoph_export():
    
    #import pdb; pdb.set_trace()

    OUTPUT_DIR = './output_2/'
    runme = [
             False, # 0 : currently empty
             False, # 1 : sandbox - preliminary creation / exporting tests
             True,  # 2 : create one instance of zoph network and save it to OUTPUT_DIR
             True,  # 3 : convert all networks in OUTPUT_DIR to onnx
             True  # 4 : load one of the saved files and check it
            ]

    shape = (10, 3, 32, 32)
    input = nn.Variable(shape)

    #  0 **************************
    if runme[0]:
        pass

    #  1 **************************
    if runme[1]:
        zn1 = zoph.SearchNet()

        zn1a_unique_active_modules = get_active_and_profiled_modules(zn1)
        zn1.save_graph('./sandbox/zn1a')
        zn1.save_modules_nnp('./sandbox/zn1a', active_only=True)
        with open('./sandbox/zn1a.txt', 'w') as f:
            print_me(zn1, f)

        out1b = zn1(input)
        zn1b_unique_active_modules = get_active_and_profiled_modules(zn1)
        zn1.save_graph('./sandbox/zn1b')
        zn1.save_modules_nnp('./sandbox/zn1b', active_only=True)
        with open('./sandbox/zn1b.txt', 'w') as f:
            print_me(zn1, f)
    
        out1c = zn1(input)
        zn1c_unique_active_modules = get_active_and_profiled_modules(zn1)
        zn1.save_graph('./sandbox/zn1c')
        zn1.save_modules_nnp('./sandbox/zn1c', active_only=True)
        with open('./sandbox/zn1c.txt', 'w') as f:
            print_me(zn1, f)
    
    #  2 **************************
    if runme[2]:
        # Sample one ZOPH search space
        zn = zoph.SearchNet()
        output = zn(input)

        zn_unique_active_modules = get_active_and_profiled_modules(zn)

        zn.save_graph      (OUTPUT_DIR + 'zn')
        zn.save_net_nnp    (OUTPUT_DIR + 'zn', input, output)
        zn.save_modules_nnp(OUTPUT_DIR + 'zn', active_only=True)
        
        with open(OUTPUT_DIR + 'zn.txt', 'w') as f:
            print_me(zn, f)

    #  3 **************************
    if runme[3]:
        zn.convert_npp_to_onnx(OUTPUT_DIR)

    #  4 **************************
    if runme[4]:
        #nnp = load(filename)
        #net = nnp.get_network(nnp.get_network_names()[0])

        nnp = load('./output/zn_whole_net.nnp')
        net_name = nnp.get_network_names()[0]
        net = nnp.get_network(net_name)


if __name__ == '__main__':
    zoph_export()