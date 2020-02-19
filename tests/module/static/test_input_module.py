import nnabla as nn
from nnabla_nas.module import static as smo

def test_input_module():
    shape = (10,3,32,32)
    input = nn.Variable(shape)
     
    inp_module = smo.Input(value=input)

    assert inp_module() == input
    assert inp_module.shape == shape

    inp_module.reset_value()
    assert inp_module._value == input

if __name__=='__main__':
    test_input_module() 
