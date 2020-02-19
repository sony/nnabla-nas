import nnabla as nn
from nnabla_nas.module import static as smo

def test_global_avgpool_module():
    shape = (10,3,32,32)
    input = smo.Input(nn.Variable(shape))
     
    inp_module = smo.Input(value=input)
    pool = smo.GlobalAvgPool(parents=[inp_module])
    assert pool.shape == (10,3,1,1)


if __name__=='__main__':
    test_global_avgpool_module() 
