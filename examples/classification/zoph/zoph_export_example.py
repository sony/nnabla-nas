
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


def get_modules(zn):
    list = []
    for mi in zn.get_net_modules(active_only=True):
        if type(mi) in zn.modules_to_profile:
            if type(mi) not in list:
                print(type(mi))        
                list.append(type(mi)) 
    print('\n********************************\n')
    return list

def zoph_export():
    
    shape = (10, 3, 32, 32)
    input = nn.Variable(shape)

    zn1 = zoph.SearchNet()

    with open('zn1a.txt', 'w') as f:
        print_me(zn1, f)
    zn1a_unique_active_modules = get_modules(zn1)
    #import pdb; pdb.set_trace()

    out1b = zn1(input)
    with open('zn1b.txt', 'w') as f:
        print_me(zn1, f)
    zn1b_unique_active_modules = get_modules(zn1)
    
    #import pdb; pdb.set_trace()
    
    out1c = zn1(input)
    with open('zn1c.txt', 'w') as f:
        print_me(zn1, f)
    zn1c_unique_active_modules = get_modules(zn1)

    #import pdb; pdb.set_trace()

    zn2 = zoph.SearchNet()
    out2 = zn2(input)
    with open('zn2.txt', 'w') as f:
        print_me(zn2, f)
    zn2_unique_active_modules = get_modules(zn2)


    zn3 = zoph.SearchNet()
    out3 = zn3(input)
    with open('zn3.txt', 'w') as f:
        print_me(zn3, f)
    zn3_unique_active_modules = get_modules(zn3)


    #zn4 = zoph.SearchNet()
    #out4 = zn4(input)

    #zn5 = zoph.SearchNet()
    #out5 = zn5(input)

    #import pdb; pdb.set_trace()    
    
    #zn1.save_modules_nnp('.', True)
    #zn1.save('.')
    
    



if __name__ == '__main__':
    zoph_export()