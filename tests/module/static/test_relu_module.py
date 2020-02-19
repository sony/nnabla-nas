import nnabla as nn
from nnabla_nas.module import static as smo

def test_relu_module():
    input = smo.Input(value=nn.Variable((8, 5)))
    relu = smo.ReLU(parents=[input])
    output = relu()
    
    assert output.shape == (8, 5)


if __name__=='__main__':
    test_relu_module() 
