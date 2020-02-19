import nnabla as nn
from nnabla_nas.module import static as smo

def test_collapse_module():
    shape = (10,3,1,1)
    input = smo.Input(nn.Variable(shape))
     
    inp_module = smo.Input(value=input)
    collapse = smo.Collapse(parents=[inp_module])
 
    assert collapse.shape == (10, 3)


if __name__=='__main__':
    test_collapse_module() 
