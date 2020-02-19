import nnabla as nn
from nnabla_nas.module import static as smo

def test_dropout_module():
    input = smo.Input(value=nn.Variable((8, 5)))
    dropout = smo.Dropout(parents=[input])
    output = dropout()
    
    assert output.shape == (8, 5)


if __name__=='__main__':
    test_dropout_module() 
