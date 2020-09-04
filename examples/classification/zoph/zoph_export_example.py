
import nnabla as nn

from nnabla_nas.contrib import zoph

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
    
    shape = (10, 3, 32, 32)
    input = nn.Variable(shape)

    #import pdb; pdb.set_trace()

    runme = [False, False, True, False]
    
    # ZN 0 **************************
    if runme[0]:
        pass

    # ZN 1 **************************
    if runme[1]:
        zn1 = zoph.SearchNet()
        with open('zn1a.txt', 'w') as f:
            print_me(zn1, f)
        zn1a_unique_active_modules = get_active_and_profiled_modules(zn1)
        zn1.save('./graphs/zn1a')
        zn1.save_modules_nnp('./graphs/zn1a', active_only=True)

        out1b = zn1(input)
        with open('zn1b.txt', 'w') as f:
            print_me(zn1, f)
        zn1b_unique_active_modules = get_active_and_profiled_modules(zn1)
        zn1.save('./graphs/zn1b')
        zn1.save_modules_nnp('./graphs/zn1b', active_only=True)
    
        out1c = zn1(input)
        with open('zn1c.txt', 'w') as f:
            print_me(zn1, f)
        zn1c_unique_active_modules = get_active_and_profiled_modules(zn1)
        zn1.save('./graphs/zn1c')
        zn1.save_modules_nnp('./graphs/zn1c', active_only=True)
    
    # ZN 2 **************************
    if runme[2]:
        zn2 = zoph.SearchNet()
        out2 = zn2(input)
        with open('zn2.txt', 'w') as f:
            print_me(zn2, f)
        zn2_unique_active_modules = get_active_and_profiled_modules(zn2)
        zn2.save('./graphs/zn2')
        zn2.save_modules_nnp('./graphs/zn2', active_only=True)

    # ZN 3 **************************
    if runme[3]:
        zn3 = zoph.SearchNet()
        out3 = zn3(input)
        with open('zn3.txt', 'w') as f:
            print_me(zn3, f)
        zn3_unique_active_modules = get_active_and_profiled_modules(zn3)
        zn3.save('./graphs/zn3')
        zn3.save_modules_nnp('./graphs/zn3', active_only=True)
    








    

if __name__ == '__main__':
    zoph_export()