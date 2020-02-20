import nnabla as nn
import nnabla.functions as F
from nnabla_nas.module.parameter import Parameter
from nnabla_nas.module import static as smo
from nnabla_nas import module as mo

def test_graph_module():
    class MyGraph(smo.Graph):
        def __init__(self, parents):
            smo.Graph.__init__(self, parents=parents)
            join_param = Parameter(shape=(len(parents)+3,))
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
                                 parents=parents + [mi for mi in self],
                                 mode='linear',
                                 join_parameters=join_param))

    
    input = smo.Input(value=nn.Variable((10, 20, 32, 32)))

    my_graph = MyGraph(parents=[input])
    output = my_graph()
    assert output.shape == (10, 20, 32, 32)
 
    mods = [mi for _, mi in my_graph.get_modules()]
    assert len(mods) == 5

if __name__=='__main__':
    test_graph_module() 
