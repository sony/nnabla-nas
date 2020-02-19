import nnabla as nn
from nnabla_nas.module import static as smo

def test_identity_module():
    shape = (10,3,32,32)
    input = nn.Variable(shape)
     
    inp_module = smo.Input(value=input)
    identity = smo.Identity(parents=[inp_module])
    
    assert identity() == input

if __name__=='__main__':
    test_identity_module() 
