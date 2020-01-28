import nnabla as nn
from nnabla_nas.module import *

class ConvBnRelu6(StaticConv):
    def __init__(self, *args, **kwargs):
        super(ConvBnRelu6, self).__init__(*args, **kwargs)
        self._bn_module = BatchNormalization(self.parent[0].shape[1], 4)
        self._relu_module = ReLU(inplace=True)

    def _value_function(self, input):
        conv_out = StaticConv._value_function(self, input)
        bn_out = self._bn_module(conv_out)
        return self._relu_module(bn_out)

