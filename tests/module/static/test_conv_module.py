import nnabla as nn
from nnabla_nas.module import static as smo

def test_conv_module():
    shape = (10,3,32,32)
    input = smo.Input(nn.Variable(shape))
     
    inp_module = smo.Input(value=input)
    conv = smo.Conv(parents=[inp_module],
                    in_channels=3,
                    out_channels=64,
                    kernel=(3,3))
 
    assert conv.shape == (10,64,30,30)


if __name__=='__main__':
    test_conv_module() 
