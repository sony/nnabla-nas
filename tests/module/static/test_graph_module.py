import nnabla as nn
import nnabla.functions as F
from nnabla_nas.module.parameter import Parameter
from nnabla_nas.module import static as smo
from nnabla_nas import module as mo

def test_graph_module():
    class MyGraph(smo.Graph):
        def __init__(self):
            smo.Graph.__init__(self)
            join_param = Parameter(shape=(3,))
            join_prob = F.softmax(join_param)
            self.append(smo.Input(name='input_1',
                                  value=nn.Variable((10, 20, 32, 32)),
                                  eval_prob=join_prob[0]+join_prob[1]))
            self.append(smo.Conv(name='conv', 
                                 parents=[self[-1]], 
                                 in_channels=20,
                                 out_channels=20, kernel=(3, 3), pad=(1, 1),
                                 eval_prob=join_prob[0]))
            self.append(smo.Input(name='input_2',
                                  value=nn.Variable((10, 20, 32, 32)),
                                  eval_prob=join_prob[2]))
            self.append(smo.Join(name='join',
                                 parents=self,
                                 mode='linear',
                                 join_parameters=join_param))

    
    my_graph = MyGraph()
    #output = my_graph()
    import pdb; pdb.set_trace() 

if __name__=='__main__':
    test_graph_module() 
