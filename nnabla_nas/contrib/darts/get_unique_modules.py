import nnabla as nn
from nnabla_nas.contrib.darts import SearchNet

net = SearchNet(3, 36, 15, 10)
input = nn.Variable((1,3,32,32))

out = net(input)

mods = [mi for _,mi in net.get_modules() if len(mi.modules) == 0]

def uid(module):
   return str(type(module)) + str(module.inputs[0])

unique_mods = {}
for mi in mods:
   #inpi = mi.inputs[0]
   #outi = mi(inpi)

   unique_mods[uid(mi)] = mi    



import pdb;pdb.set_trace()
 
