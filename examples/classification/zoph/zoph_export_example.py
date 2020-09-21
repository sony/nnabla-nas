
import os
import nnabla as nn



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

def export_all():
    
    #import pdb; pdb.set_trace()

    runme = [
             False, # 0 : sandbox - preliminary creation / exporting tests
             False, # 1 : create one instance of ZOPH network, save it and its modules, convert them to ONNX
             False,  # 2 : sample a set of ZOPH networks and export all of them (but just the whole net not the modules), export them to ONNX
             True,  # 3 : random wired search space (static).  create one instance of ZOPH network, save it and its modules, convert to ONNX
             True,  # 4 :  sample a set of RANDOM WIRED networks and export all of them (but just the whole net not the modules), export them to ONNX
             False, # 5 : testing the export for dynamic modules
            ]


    #  0 **************************
    if runme[0]:
        from nnabla_nas.contrib import zoph

        shape = (10, 3, 32, 32)
        input = nn.Variable(shape)
        
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

    #  1 **************************
    if runme[1]:
        from nnabla_nas.contrib import zoph

        OUTPUT_DIR = './output_1/'

        # Sample one ZOPH network from the search space
        shape = (10, 3, 32, 32)
        input = nn.Variable(shape)
        zn = zoph.SearchNet()
        output = zn(input)

        #zn_unique_active_modules = get_active_and_profiled_modules(zn)

        zn.save_graph      (OUTPUT_DIR + 'zn')
        zn.save_net_nnp    (OUTPUT_DIR + 'zn', input, output)
        zn.save_modules_nnp(OUTPUT_DIR + 'zn', active_only=True)
        zn.convert_npp_to_onnx(OUTPUT_DIR)

        with open(OUTPUT_DIR + 'zn.txt', 'w') as f:
            print_me(zn, f)
    

    #  2 **************************
    if runme[2]:
        from nnabla_nas.contrib import zoph

        OUTPUT_DIR = './output_2/'
        
        shape = (10, 3, 32, 32)
        input = nn.Variable(shape)
        N = 10  # number of random networks to sample

        # Sample N zoph networks from the search space
        for i in range(0,N):
            zn = zoph.SearchNet()
            output = zn(input)
            zn.save_net_nnp    (OUTPUT_DIR + 'zn' + str(i), input, output)
        zn.convert_npp_to_onnx(OUTPUT_DIR)

    #  3 **************************
    if runme[3]:
        from nnabla_nas.contrib import random_wired

        OUTPUT_DIR = './output_3/'

        # Sample one random wired network from the search space
        shape = (10, 3, 32, 32)
        input = nn.Variable(shape)
        rw = random_wired.TrainNet()
        output = rw(input)
        
        rw.save_graph      (OUTPUT_DIR + 'rw')
        rw.save_net_nnp    (OUTPUT_DIR + 'rw', input, output)
        rw.save_modules_nnp(OUTPUT_DIR + 'rw', active_only=True)
        rw.convert_npp_to_onnx(OUTPUT_DIR)


    #  4 **************************
    if runme[4]:
        from nnabla_nas.contrib import random

        OUTPUT_DIR = './output_4/'
        
        shape = (10, 3, 32, 32)
        input = nn.Variable(shape)
        N = 10  # number of random networks to sample

        # Sample N zoph networks from the search space
        for i in range(0,N):
            rw = random_wired.TrainNet()
            output = rw(input)
            rw.save_net_nnp    (OUTPUT_DIR + 'zn' + str(i), input, output)
        zn.convert_npp_to_onnx(OUTPUT_DIR)

    #  5 **************************
    if runme[5]:
        from nnabla_nas.contrib.classification.mobilenet import SearchNet

        OUTPUT_DIR = './output_5/'
        
        mobile_net = SearchNet(num_classes=1000)
        input = nn.Variable((1, 3, 224, 224))
        output = mobile_net(input)
        
        #mobile_net.save_modules_nnp(OUTPUT_DIR + 'mn', True)

        # mobile_net.save_net_nnp(OUTPUT_DIR + 'mn', input, output)
        # it seems the last affine layer cannot be converted to ONNX (!?),
        # thus export without it
        mobile_net.save_net_nnp(OUTPUT_DIR + 'mn', input, output.parent.inputs[0])
        
        mobile_net.convert_npp_to_onnx(OUTPUT_DIR)
        




if __name__ == '__main__':
    export_all()