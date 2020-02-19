import nnabla as nn
from nnabla_nas.module import static as smo

def test_zero_module():
    shape = (10,3,32,32)
    input = nn.Variable(shape)
     
    inp_module = smo.Input(value=input)
    zero = smo.Zero(parents=[inp_module])
 
    assert zero.shape == input.shape

    inp_module.value = nn.Variable((20,3,32,32))
    zero.reset_value()
    assert zero.shape == (20,3,32,32)

if __name__=='__main__':
    test_zero_module() 
